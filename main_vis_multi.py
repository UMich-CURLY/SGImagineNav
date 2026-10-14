#!/usr/bin/env python3

import os
import sys
sys.path.append(".")
sys.stdout.reconfigure(line_buffering=True)
import time
import random
import logging
import re
import json
from typing import Dict
import numpy as np
import torch
from copy import deepcopy
import shutil

from habitat import Env
from arguments import get_args
from habitat.config.default import get_config
from habitat import make_dataset
from habitat.config import read_write

from agents.objnav_agent import ObjectNav_Agent
from collections import defaultdict
import imageio

from multiprocessing import Process, Queue
import subprocess
import torch.multiprocessing as mp
#  from habitat.sims.habitat_simulator.actions import (
#     HabitatSimActions,
#     HabitatSimV1ActionSpaceConfiguration,
# )

from utils.perception_utils.gt_perception import get_gt_goal_positions, GTPerception
from utils.scene_graph_utils import get_gt_scenegraph
from utils.vis import pad_frames_to_same_size
from utils.shortest_path_follower import ShortestPathFollowerCompat
from constants import category_to_id, episode_labels_table, categories_21
import open3d.visualization.gui as gui
from utils.vis_gui import ReconstructionWindow
from utils.db_utils import connect_db, generate_schema, insert_data


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def get_scene_id(scene_path):
    # Get the scene_id from the path
    scene_id = scene_path.split("/")[-1]
    scene_id = re.sub(r'\.basis\.glb$', '', scene_id)
    return scene_id

def update_episodes(env, config, new_episodes):
    env.episodes = new_episodes
    env.number_of_episodes = len(new_episodes)
    env._dataset.episodes = new_episodes
    iter_option_dict = {
        k.lower(): v
        for k, v in config.habitat.environment.iterator_options.items()
    }
    iter_option_dict["seed"] = config.habitat.seed
    env._episode_iterator = env._dataset.get_episode_iterator(
        **iter_option_dict
    )

def run_ollama_serve(port, gpu_id):
    os.environ["OLLAMA_HOST"] = f'http://127.0.0.1:{port}'
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # torch.cuda.set_device(gpu_id)
    os_env = os.environ.copy()
    # if you use gpu_ids = "2,3", then you need to transform the gpu_id to the actual device number
    # the gpu_id here passed in is the index from 0. 
    actual_device_id = os_env["CUDA_VISIBLE_DEVICES"].split(",")[gpu_id] 
    os_env["CUDA_VISIBLE_DEVICES"] = str(actual_device_id)
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        bufsize=-1,
        env=os_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    def read_output(process, status):
        while True:
            output = process.stdout.readline()
            if b"Listening" in output:
                print("[OLLAMA] ollama serve started successfully")
                status.put(1)
            elif b"Error" in output:
                print("[OLLAMA] ", output.decode().strip())
                status.put(0)
    status = Queue()
    Process(target=read_output, args=(ollama_process, status), daemon=True).start()
    try:
        if not status.get(True, timeout=3):
            raise Exception("[OLLAMA] Error starting ollama serve")
    except Exception:
        print("[OLLAMA] Timeout waiting for ollama serve to start")
        raise
    return ollama_process

def VLNav_env(args, config, process_id, dataset, episode_labels, gui_prompt_queue, result_queue, step_data_queue, gui_viz_queue=None):
    args.process_id = process_id
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    torch.set_grad_enabled(False)

    os.environ["GPT_API_KEY"] = args.api

    # set env variables
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.set_device(args.gpu_id)
    os.environ["OLLAMA_HOST"] = args.ollama_host
    
    # set up the habitat environment
    env = Env(config, dataset)
    env.seed(config.habitat.seed)
    # env.sim.config.sim_cfg.random_seed = config.habitat.seed
    num_episodes = len(episode_labels)
    
    episodes_left = []
    for i, episode in enumerate(env.episodes):
        scene_id = get_scene_id(env.episodes[i].scene_id)
        episode_id = env.episodes[i].episode_id
        episode_label = f'{scene_id}_{episode_id}'
        if episode_label in episode_labels:
            episodes_left.append(episode)
    if len(episodes_left) == 0:
        print(f"[{args.process_label}] No episodes left to run")
        print(f"[{args.process_label}] Process finished")
        return
    # episodes_left.sort(key=lambda x: (get_scene_id(x.scene_id), int(x.episode_id)))
    update_episodes(env, config, episodes_left)
    print(f"[{args.process_label}] Total episodes: ", num_episodes)
    print(f"[{args.process_label}] First 10 episodes: ", [f'{get_scene_id(episode.scene_id)}_{episode.episode_id}' for episode in env.episodes[:10]])
    sys.stdout.flush()

    follower = ShortestPathFollowerCompat(
        env._sim, 0.1, False
    )
    
    agent = ObjectNav_Agent(args, follower)
    gt_perception = GTPerception() # to get gt object detections

    count_episodes = 0
    while count_episodes < num_episodes:
        try:
            env.seed(config.habitat.seed)
            # env.sim.config.sim_cfg.random_seed = config.habitat.seed
            obs = env.reset()

            scene_id = get_scene_id(env.current_episode.scene_id)
            episode_id = env.current_episode.episode_id
            episode_label = f'{scene_id}_{episode_id}'
            print(f"[{args.process_label}] Running episode: ", episode_label)

            agent.reset(episode_label, scene_id, episode_id)
            goal_poses_metric = get_gt_goal_positions(env.current_episode.goals,  env.sim.get_agent_state())
            obs['goal_poses_metric'] = goal_poses_metric
            
            # get ground truth info for each frame
            if args.gt_perception:
                gt_perception.set_semantic_annotaions(env.sim.semantic_annotations())
                try:
                    get_results, detections = gt_perception.detect(obs['semantic'], obs['depth'])
                except Exception as e:
                    print(f"[{args.process_label}] Error in gt detection: {e}")
                    get_results = None
                    detections = None
                
                # append gt info to observation dict
                obs['gt_detections'] = {'get_results': get_results, 'detections': detections}
            
            if args.gt_scenegraph:
                gt_scenegraph = get_gt_scenegraph(env.sim.curr_scene_name, env.sim.semantic_scene, env.sim.get_agent_state(), args.map_resolution, agent.origins_grid)
                obs['gt_scenegraph'] = gt_scenegraph
            
            video_save_path = '{}/{}/episodes_video/eps_{}_vis.mp4'.format(
                args.dump_location, args.exp_name, episode_label)
        
            count_steps = 0
            while not env.episode_over:
                agent_state = env.sim.get_agent_state()
                sim_info = {'env': env, 'scene_name': scene_id, 'episode_id': episode_id, 'agent_state': agent_state}
                action = agent.act(obs, sim_info, step_data_queue, gui_prompt_queue, gui_viz_queue)
                # debug
                # if count_steps == 30:
                #     action = 0
                if action == None:
                    continue
                obs = env.step(action)
                # get ground truth observation for each frame
                # append gt info to observation dict
                obs['goal_poses_metric'] = goal_poses_metric
                if args.gt_perception:
                    try:
                        get_results, detections = gt_perception.detect(obs['semantic'], obs['depth'])
                    except Exception:
                        # print(f"[{args.process_label}] Error in gt detection: {e}")
                        get_results = None
                        detections = None
                    obs['gt_detections'] = {'get_results': get_results, 'detections': detections}
                if args.gt_scenegraph:
                    obs['gt_scenegraph'] = gt_scenegraph
                    
                metrics = env.get_metrics()
                agent.update_metrics(metrics)
                count_steps += 1
                sys.stdout.flush()

            if action == 0 and env.get_metrics()["spl"]:
                # print("you successfully navigated to destination point")
                fail_category = 'success'
            elif count_steps >= config.habitat.environment.max_episode_steps - 1:
                fail_category = 'exploration'
            elif agent.replan_count > 20:
                fail_category = 'collision'
            else:
                fail_category = 'detection'

            metrics = env.get_metrics()

            extra_info = {
                'episode_label': episode_label,
                'objectgoal': obs['objectgoal'][0],
                'upstair_flag': agent.upstair_flag,
                'downstair_flag': agent.downstair_flag,
                'switch_upstair_count': agent.switch_upstair_count,
                'switch_downstair_count': agent.switch_downstair_count,
            }

            if args.save_video:
                same_size_frames = pad_frames_to_same_size(agent.vis_frames)
                imageio.mimsave(video_save_path, same_size_frames, fps=2)
                print(f"[{args.process_label}] Video saved to {video_save_path}")
            
            result_queue.put([metrics, fail_category, count_steps, extra_info])
            count_episodes += 1

        except Exception as e:
            count_episodes += 1
            env.step(0)
            print(f"[{args.process_label}] Error encountered: {e}")
            import traceback
            traceback.print_exc()
            if args.debug:
                raise
    print(f"[{args.process_label}] Process finished")
    sys.stdout.flush()
    env.close()


def visualization_thread(args, gui_prompt_queue, gui_viz_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app_win = ReconstructionWindow(args, mono, gui_prompt_queue, gui_viz_queue)
    app.run()
    del app_win

def save_step_data(step_data_queue, args, stop_event=None):
    sys.stdout.reconfigure(line_buffering=True)
    commit_interval = 10
    db_state_list = {}
    db_folder = f'{args.dump_location}/{args.exp_name}/steps/'
    os.makedirs(os.path.dirname(db_folder), exist_ok=True)
    get_time = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    process_label = args.process_label
    while not (stop_event.is_set() and step_data_queue.empty()):
        try:
            # get data from the queue
            if step_data_queue.empty():
                time.sleep(0.01)
                continue
            episode_steps_dict = {}
            while step_data_queue.qsize() > 0:
                step_data = step_data_queue.get()
                episode_label = step_data['episode_label']
                episode_steps_dict.setdefault(episode_label, []).append(step_data)
            # save data to db for each episode
            for episode_label, step_list in episode_steps_dict.items():
                # create a new db
                if episode_label not in db_state_list:
                    print(f"[db] [{process_label}] [{episode_label}] create {db_folder}{episode_label}_tmp.db")
                    if os.path.exists(f'{db_folder}{episode_label}_tmp.db'):
                        os.remove(f'{db_folder}{episode_label}_tmp.db')
                    con = connect_db(f'{db_folder}{episode_label}_tmp.db')
                    db_state_list[episode_label] = con
                    con.raw_sql("SET checkpoint_threshold = '100G';")
                    con.raw_sql("BEGIN TRANSACTION;")
                    # create a table according to the first data
                    schema = generate_schema(step_list)
                    con.create_table('step_data', schema=schema, overwrite=True)
                    # print(f"[db] [{process_label}] Created table step_data, schema:\n{schema}")
                    # analyze_data_size(step_list[0], prefix="")
                con = db_state_list[episode_label]
                step_data = step_list[0]
                step_id = step_data['step']
                action = step_data['action']
                # commit the transaction if required
                if (step_id>0 and step_id%commit_interval == 0):
                    print(f"[db] [{process_label}] [{episode_label}] committing transaction, time={get_time()}")
                    sys.stdout.flush()
                    start_time = time.time()
                    con.raw_sql("COMMIT;")
                    print(f"[db] [{process_label}] [{episode_label}] committed in {time.time() - start_time:.2f}s")
                    con.raw_sql("BEGIN TRANSACTION;")
                # insert data
                start_time = time.time()
                insert_data(con, 'step_data', step_list)
                print(f"[db] [{process_label}] [{episode_label}] Inserted {len(step_list)} step data in {time.time() - start_time:.2f}s, step={step_id}, time={get_time()}")
                # commit, checkpoint and close the db
                if step_id == 499 or action == 0:
                    print(f"[db] [{process_label}] [{episode_label}] episode end, committing and checkpointing, time={get_time()}")
                    con.raw_sql("COMMIT;")
                    start_time = time.time()
                    con.raw_sql("CHECKPOINT;")
                    print(f"[db] [{process_label}] [{episode_label}] checkpointed in {time.time() - start_time:.2f}s")
                    con.disconnect()
                    # move tmp db to final db
                    db_path = f'{db_folder}{episode_label}.db'
                    if os.path.exists(db_path):
                        os.remove(db_path)
                    print(f"[db] [{process_label}] [{episode_label}] moving {db_folder}{episode_label}_tmp.db to {db_path}")
                    os.rename(f'{db_folder}{episode_label}_tmp.db', db_path)
        except Exception as e:
            print(f"[db] [{process_label}] Error encountered: {e}")
            import traceback
            traceback.print_exc()
            if args.debug:
                raise
        sys.stdout.flush()


def update_result(result_queue, benchmark_state, agg_metrics, start_time, log_dir, total_fail, per_episode_error, count_episode_left, args):
    while True:
        if not result_queue.empty():
            print("[update_result] Yay! New result received!")
            metrics, fail_category, count_steps, extra_info = result_queue.get()
            episode_label = extra_info['episode_label']

            benchmark_state["count_episodes"] += 1
            benchmark_state["total_steps"] += count_steps
            total_steps = benchmark_state["total_steps"]
            count_episodes = benchmark_state["count_episodes"]
            
            for m, v in metrics.items():
                agg_metrics[m] += v
                
            total_fail[fail_category] += 1

            end_time = time.time()
            time_elapsed = time.gmtime(end_time - start_time)
            log = " ".join([
                "Episode: {}".format(episode_label),
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(total_steps ),
                "FPS {},".format(int(total_steps  / (end_time - start_time)))
            ]) + '\n'

            log += f"Failed Case: collision={total_fail['collision']}, exploration={total_fail['exploration']}, detection={total_fail['detection']}, success={total_fail['success']}, total={len(per_episode_error)}\n"
            
            log += "Metrics: "
            log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, count_episode_left)

            print(log)
            logging.info(log)

            case_summary = {}
            case_summary["episode"] = episode_label
            case_summary["habitat_success"] = metrics["success"]
            case_summary['distance_to_goal'] = metrics['distance_to_goal']
            case_summary['spl'] = metrics['spl']
            case_summary['soft_spl'] = metrics['soft_spl']
            case_summary['success'] = total_fail['success']
            case_summary['exploration'] = total_fail['exploration']
            case_summary['collision'] = total_fail['collision']
            case_summary['detection'] = total_fail['detection']
            case_summary['fail_category'] = fail_category
            case_summary['switch_upstair_count'] = extra_info['switch_upstair_count']
            case_summary['switch_downstair_count'] = extra_info['switch_downstair_count']
            case_summary["upstair_flag"] = extra_info['upstair_flag']
            case_summary["downstair_flag"] = extra_info['downstair_flag']
            case_summary["count_steps"] = count_steps
            if args.task_config == 'objectnav_mp3d_rgbd.yaml':
                category_list = categories_21 # mp3d has 21 classes
            else:
                category_list = category_to_id # hm3d, ai2thor, hssd all has similar target
            case_summary["target"] = category_list[extra_info['objectgoal']]
            per_episode_error.append(case_summary)
            with open(log_dir + "all_info.log", 'w') as fp:
                for item in per_episode_error:
                    # write each item on a new line
                    fp.write(json.dumps(item) + "\n")
            # save in database
            step_db_path = f'{args.dump_location}/{args.exp_name}/result.db'
            con = connect_db(step_db_path)
            if not 'result' in con.list_tables():
                schema = generate_schema(case_summary)
                con.create_table('result', schema=schema, overwrite=True)
                print(f"[update_result] Created table result, schema:\n{schema}")
            insert_data(con, 'result', case_summary)
            con.disconnect()
        time.sleep(0.1)
        sys.stdout.flush()

def main():
    print(f"VISIBLE DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    args = get_args()

    # logging
    args.exp_name = args.exp_name + "-" + args.detector
    log_dir = "{}/{}/logs/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    video_save_dir = '{}/{}/episodes_video'.format(
                args.dump_location, args.exp_name)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)
    public_args = vars(args).copy()
    public_args["api"] = "[REDACTED]"
    logging.info(public_args)

    # copy monitor html
    if not os.path.exists('{}/{}/monitor.html'.format(args.dump_location, args.exp_name)):
        shutil.copy("tools/monitor.html", '{}/{}/monitor.html'.format(args.dump_location, args.exp_name))

    # save args and config files
    with open(args.dump_location + "/args.json", 'w') as fp:
        json.dump(public_args, fp, indent=4)
    shutil.copy("exp_configs/"+ args.exp_config, args.dump_location)

    # multiprocessing
    mp_ctx = mp.get_context("forkserver")
    gui_prompt_queue = mp_ctx.Queue() # query from the gui
    result_queue = mp_ctx.Queue() # query for gathering metrics
    gui_viz_queue = mp_ctx.Queue() # data for visualization
    step_data_queue_list = []
    stop_event = mp_ctx.Event()

    # load config
    # config_env = get_config(config_paths=["configs/" + args.task_config])
    config_env = get_config("configs/" + args.task_config)

    with read_write(config_env):
        # config_env.defrost()
        config_env.habitat.dataset.split = args.split
        config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_id
        # config_env.freeze()

    scenes = config_env.habitat.dataset.content_scenes
    dataset = make_dataset(config_env.habitat.dataset.type, config=config_env.habitat.dataset)
    if "*" in config_env.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(config_env.habitat.dataset)
    assert len(scenes) > 0, "Error: failed to load scenes"

    # read all_info.log
    per_episode_error = []
    if os.path.exists(log_dir + "all_info.log"):
        with open(log_dir + "all_info.log", 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line != "":
                    per_episode_error.append(json.loads(line))
    episode_done_list = [x["episode"] for x in per_episode_error]
    print("{} episode info from all_info.log is loaded".format(len(per_episode_error)))
    
    total_fail = {'success': 0, 'exploration': 0, 'collision': 0, 'detection': 0}
    if len(per_episode_error) != 0:
        last_log = per_episode_error[-1]
        for key in total_fail.keys():  
            total_fail[key] = last_log[key]

    # set processes and gpu
    num_gpu = args.num_gpu
    num_processes = args.num_processes
    if num_gpu == 0:
        num_gpu = torch.cuda.device_count()
    num_all_processes = num_processes * num_gpu
    print("Number of all threads: ", num_all_processes)
    print("Number of GPUs: ", num_gpu)
    print("Number of processes on each GPU: ", num_processes)

    # load all episodes
    proc_config = deepcopy(config_env)
    with read_write(proc_config):
        # proc_config.defrost()
        proc_config.habitat.dataset.split = args.split
        proc_config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_id
        dataset = make_dataset(proc_config.habitat.dataset.type, config=proc_config.habitat.dataset)
        proc_config.habitat.simulator.scene = dataset.episodes[0].scene_id
        # proc_config.freeze()
    env = Env(proc_config, dataset)
    
    if args.debug:
        # save the config file
        from omegaconf import OmegaConf
        data = OmegaConf.to_yaml(proc_config) # yaml format string
        with open('task_config.yml', 'w') as outfile:
            outfile.write(data)
        # raise ValueError("Debug mode is on, please check the config file.")

    # load all episodes
    episode_labels = []
    for i in range(len(env.episodes)):
        scene_id = get_scene_id(env.episodes[i].scene_id)
        episode_id = env.episodes[i].episode_id
        episode_label = f'{scene_id}_{episode_id}'
        episode_labels.append(episode_label)
    env.close()

    # filter the episodes
    if args.episode_labels is not None and args.episode_labels != "":
        if args.episode_labels[0]=='!':
            episode_labels = [x for x in episode_labels if x not in episode_labels_table[args.episode_labels]]
        elif args.episode_labels in episode_labels_table:
            episode_labels = episode_labels_table[args.episode_labels]
        else:
            episode_labels = [args.episode_labels]

    print(f"Total number of episodes: {len(episode_labels)}")

    # remove repeated episodes
    unique_episode_labels = set(episode_labels)
    print(f"Remove {len(episode_labels) - len(unique_episode_labels)} repeated episodes")
    episode_labels = list(unique_episode_labels)
    new_per_episode_error = []
    added_labels = []
    for episode_error in per_episode_error:
        if episode_error["episode"] not in added_labels:
            new_per_episode_error.append(episode_error)
            added_labels.append(episode_error["episode"])
    per_episode_error = new_per_episode_error

    # skip episodes if video already exists
    print(f"Check existing videos...")
    count_episode_done = 0
    count_episode_left = 0
    episode_labels_left = []
    remove_log_list = []
    for episode_label in episode_labels:
        video_save_path = '{}/{}/episodes_video/eps_{}_vis.mp4'.format(args.dump_location, args.exp_name, episode_label)
        if os.path.exists(video_save_path):
            if episode_label in episode_done_list:
                # print(f"Episode {episode_label} found in all_info.log.")
                count_episode_done += 1
            else:
                print(f"Episode {episode_label} is missing in all_info.log. Video file removed.")
                os.remove(video_save_path)
                count_episode_left += 1
                episode_labels_left.append(episode_label)
        else:
            if episode_label in episode_done_list:
                print(f"Episode video {episode_label} is missing.")
                count_episode_done += 1
                continue
                # print(f"Episode video {episode_label} is missing. Result is removed from all_info.log")
                # remove_log_list.append(episode_label)
            count_episode_left += 1
            episode_labels_left.append(episode_label)
    episode_labels = episode_labels_left
    print(f"Remove {len(remove_log_list)} episodes from all_info.log")
    per_episode_error = [x for x in per_episode_error if x["episode"] not in remove_log_list]
    with open(log_dir + "all_info.log", 'w') as f:
        for item in per_episode_error:
            f.write(json.dumps(item) + "\n")
    # episode_labels.sort(key=lambda x: (x.split("_")[0], int(x.split("_")[1])))

    print(f"Number of episodes done: {count_episode_done}")
    print(f"Number of episodes left: {count_episode_left}")
    print(f"Check with all_info.log: {len(per_episode_error)} (all_info.log)=={count_episode_done} (script)")

    # split the scenes into multiple processes
    split_sizes = [len(episode_labels) // num_all_processes] * num_all_processes
    for i in range(len(episode_labels) % num_all_processes):
        split_sizes[i] += 1
    print("Episodes per process:")
    for gpu_id in range(num_gpu):
        for i in range(num_processes):
            process_id = i*num_gpu+gpu_id
            print(f'gpu_id: {gpu_id}, process: {i}, n_episode: {split_sizes[process_id]}')

    # define result update function
    benchmark_state = {
        "count_episodes": 0,
        "total_steps": 0,
    }
    agg_metrics: Dict = defaultdict(float)
    start_time = time.time()

    # start the processes
    # visualization
    if args.visualize:
        # Create a process for the Open3D visualization
        print("Start visualization process")
        visualization = mp_ctx.Process(target=visualization_thread, args=(args, gui_prompt_queue, gui_viz_queue))
        visualization.daemon = True
        visualization.start()
    # result gathering
    print("Start result gathering process")
    result_gathering = mp_ctx.Process(target=update_result, args=[result_queue, benchmark_state, agg_metrics, start_time, log_dir, total_fail, per_episode_error, count_episode_left, args])
    result_gathering.daemon = True
    result_gathering.start()
    # vln
    processes = []
    ollama_processes = []
    ollama_host = args.ollama_host
    for gpu_id in range(num_gpu):
        for i in range(num_processes):
            process_id = i*num_gpu+gpu_id

            # create config
            proc_config = deepcopy(config_env)
            with read_write(proc_config):
                # proc_config.defrost()
                proc_config.habitat.dataset.split = args.split

                if num_all_processes==1:
                    proc_config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu_id
                else:
                    args.gpu_id = gpu_id
                    proc_config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
                args.process_label = f'gpu{args.gpu_id}_p{i}'

                dataset = make_dataset(proc_config.habitat.dataset.type, config=proc_config.habitat.dataset)
                # proc_config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
                # proc_config.freeze()

            start_episode = sum(split_sizes[:process_id])
            end_episode = sum(split_sizes[:process_id+1])
            episode_labels_process = episode_labels[start_episode:end_episode]
            if len(episode_labels_process) == 0:
                print(f"[{args.process_label}] No episodes left to run")
                continue
            print(f"[{args.process_label}] First 10 episodes: {episode_labels_process[:10]}")

            # run ollama
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            # torch.cuda.set_device(args.gpu_id)
            
            ollama_launch_success = False
            if args.imagine_nav_planner:
                while not ollama_launch_success:
                    ollama_port = args.ollama_port_start + process_id
                    print(f"[{args.process_label}] Starting ollama serve on port {ollama_port}")
                    try:
                        ollama_process = run_ollama_serve(ollama_port, args.gpu_id)
                        ollama_launch_success = True
                    except Exception:
                        print(f"[{args.process_label}] OLLAMA PORT {ollama_port} already in use, retrying...")
                        args.ollama_port_start += 100
                ollama_processes.append(ollama_process)
                args.ollama_host = f"{ollama_host}:{ollama_port}"
            
            # create step data process
            step_data_queue = mp_ctx.Queue()
            step_data_queue_list.append(step_data_queue)
            if args.save_step_data:
                print("Start step data saving process")
                step_data_process = mp_ctx.Process(target=save_step_data, args=[step_data_queue, args, stop_event])
                step_data_process.daemon = True
                step_data_process.start()

            # create and start process
            if num_all_processes==1:
                ##### NOTE: overwrite the episode labels for stair climbing episode #####
                if args.debug and args.task_config == 'objectnav_hm3d_rgbd_semantic.yaml' or args.task_config == 'objectnav_hm3d_rgbd.yaml':
                    # load upstair episodes
                    file_name = "experiments/hm3d_upstair_episode_id.json"
                    episode_labels_process = []
                    with open(file_name) as json_file:
                        floor_episode_dict = json.load(json_file)
                    for key, val in floor_episode_dict.items():
                        episode_labels_process += val[:1]
                    # load downstair episodes
                    file_name = "experiments/hm3d_downstair_episode_id.json"
                    episode_labels_process = []
                    with open(file_name) as json_file:
                        floor_episode_dict = json.load(json_file)
                    for key, val in floor_episode_dict.items():
                        episode_labels_process += val[:1]
                    # episode_labels_process = ["cvZr5TUy5C5_62", "4ok3usBNeis_91", "mL8ThkuaVTM_36", "cvZr5TUy5C5_65", "cvZr5TUy5C5_32"]
                    # svBbv1Pavdk_39 black hole stuck
                    # svBbv1Pavdk_91 black hole stuck always replanning
                    # cvZr5TUy5C5_70 repeat look up and look down so stuck
                    # cvZr5TUy5C5_74 spawn directly on stair
                    # 6s7QHgap2fW_52 find bed upstair, stair is easy to see XB4GS9ShBRE_46
                    # QaLdnwvtxbs_53 annotated with region label
                    # 5cdEh9F2hJL_61 goal in obstacle
                    # episode_labels_process = ["cvZr5TUy5C5_62", "4ok3usBNeis_91", "mL8ThkuaVTM_36", "cvZr5TUy5C5_65", "cvZr5TUy5C5_32"]
                    # QaLdnwvtxbs_78, Dd4bFSTQ8gi_75, DYehNKdT76V_18,q3zU7Yy5E5s_22qyAac8rV8Zk_34
                    episode_labels_process = ["cvZr5TUy5C5_62"] #6s7QHgap2fW_48
                    # episode_labels_process=['TEEsavR23oF_38', 'Dd4bFSTQ8gi_75', 'TEEsavR23oF_11', 'mL8ThkuaVTM_91', 'q3zU7Yy5E5s_23', 'cvZr5TUy5C5_65', 'q3zU7Yy5E5s_22', 'ziup5kvtCCR_30', 'q3zU7Yy5E5s_41', 'QaLdnwvtxbs_76', 'QaLdnwvtxbs_78', 'svBbv1Pavdk_65']

                ##### NOTE: overwrite the episode labels for stair climbing episode #####
                VLNav_env(args, proc_config, process_id, dataset, episode_labels_process, gui_prompt_queue, result_queue, step_data_queue, gui_viz_queue)
                break
            if i==0 and gpu_id==0:
                # send the actual visualization queue to the first process
                proc = mp_ctx.Process(target=VLNav_env, args=(args, proc_config, process_id, dataset, episode_labels_process, gui_prompt_queue, result_queue, step_data_queue, gui_viz_queue))
            else:
                # send a dummy send queue
                gui_prompt_queue_dummy = mp_ctx.Queue()
                proc = mp_ctx.Process(target=VLNav_env, args=(args, proc_config, process_id, dataset, episode_labels_process, gui_prompt_queue_dummy, result_queue, step_data_queue))
            proc.daemon = True
            processes.append(proc)
            start_success = False
            while not start_success:
                try:
                    proc.start()
                    start_success = True
                except Exception as e:
                    print(f"Error starting process {i} on GPU {gpu_id}: {e}, retrying...")
                    time.sleep(1)
            print(f"VLN process {i} on GPU {gpu_id} started")

    print("All processes started!")

    # wait for all agent processes to finish
    for proc in processes:
        proc.join()
    # wait for result gathering process to finish
    time.sleep(10)
    result_gathering.terminate()
    stop_event.set()
    if args.save_step_data:
        step_data_process.join()
    if args.visualize:
        visualization.close()
    for ollama_process in ollama_processes:
        ollama_process.terminate()
    print("All processes finished!")

if __name__ == "__main__":
    main()
