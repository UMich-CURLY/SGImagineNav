import argparse
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Visual-Language-Navigation')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    
    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('-d', '--dump_location', type=str, default="./dump",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="objectnav",
                        help='experiment name (default: exp1)')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', action="store_true", default=True,
                        help='save visualization as images')
    parser.add_argument('--save_scene_graph', action="store_true", default=False, 
                        help='save scene graph')
    parser.add_argument('--save_video', action="store_true", default=True,
                        help='save visualization as video')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('-n', '--num_processes', type=int, default=1, 
                        help='number of processes to use for each gpu')
    parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use, 0 for all available GPUs')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--episode_labels', type=str, default=None, help='use specific episode label or key defined in constants')
    # gui
    parser.add_argument('--path_npz', type=str, default="./saved_pcd/",
                        help='path to saved pcd (default: ./saved_pcd/)')

    
    # Environment, dataset and episode specifications
    parser.add_argument("--task_config", type=str,
                        # default="objectnav_hm3d.yaml",
                        # default='objectnav_hm3d_semantic_only.yaml',
                        # default='objectnav_hm3d_new.yaml',
                        default='objectnav_hm3d_rgbd_semantic.yaml',
                        help="path to config yaml containing task information")
    parser.add_argument("--exp_config", type=str,
                        default="Utility_ObjectRegion_ExploitationOnly_Prediction.yaml", #Utility_ObjectOnly_ExploitationOnly_ObservedOnly #Utility_ObjectRegion_ExploitationOnly_Prediction
                        help="path to config yaml containing scene graph information")
    parser.add_argument('--episode_count', type=int, default=-1)
    parser.add_argument('--split', type=str, default="val")

    # Model Hyperparameters
    parser.add_argument('--turn_angle', type=int, default=30)
    
    # Mapping
    parser.add_argument('-fw', '--frame_width', type=int, default=640,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=480,
                        help='Frame height (default:120)')
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--min_depth', type=float, default=0.0,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--map_height_cm', type=int, default=110)
    parser.add_argument('--collision_threshold', type=float, default=0.10)
    parser.add_argument('--temporal_collision', action="store_true",
                        help="if set, the temporal collision will be used")


    # SAM setting
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    parser.add_argument("--sam_variant", type=str, default="sam",
                        choices=['fastsam', 'mobilesam', "lighthqsam", "sam"])
    parser.add_argument("--detector", type=str, default="dino", 
                        choices=["yolo", "dino", "none"], 
                        help="If none, no tagging and detection will be used and the SAM will be run in dense sampling mode. ")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="if set, the class set will be accumulated over frames")


    # LLM setting
    parser.add_argument('--vln_mode', type=str, default="vlm_game",
                        choices=['clip', 'vlm', 'vlm_rank', "vlm_game"])
    # parser.add_argument('--gpt_type', type=int, default=2,
    #                     help="""0: text-davinci-003
    #                             1: gpt-3.5-turbo
    #                             2: gpt-4o
    #                             3: gpt-4o-mini
    #                             (default: 2)""")
    parser.add_argument('--api', type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help='API key; prefer the OPENAI_API_KEY environment variable')

    parser.add_argument('--scene_graph_prediction_llm', type=str, default="gpt-4o-mini")
    parser.add_argument('--group_caption_vlm', type=str, default="gpt-4o-mini") # gpt-4o-mini, llama3.2-vision
    parser.add_argument('--ollama_host', type=str, default="http://localhost", help='do not add port, it will be added automatically')
    parser.add_argument('--ollama_port_start', type=int, default="11888", help='every process will use a different port ollama_port_start+i')

    parser.add_argument('--load', type=str, default="0",
                    help="""model path to load,
                            0 to not reload (default: 0)""")
    parser.add_argument('--llm_reperception', action="store_true",
                        help="if set, the llm will be used to re-perceive the objects")
    parser.add_argument('--reperception_llm', type=str, default="llama3.2-vision")
    
    # planning 
    parser.add_argument('--fmm_planner', action="store_true", default=True,
                        help="if set, the planner will use the Fast Marching Method")
    parser.add_argument('--imagine_nav_planner', action="store_true",
                        help="if set, the planner will use the Fast Marching Method")
    parser.add_argument('--no_llm', action="store_true", default=False,
                        help="if set, disable region caption and scene graph prediction")
    parser.add_argument('--no_deal_stair_spawn', action="store_true",
                        help="if set, the agent will not deal with case when spawn on stairs")
    parser.add_argument('--early_climb', action="store_true",
                        help="if set, the agent will climb stairs earlier according to the criteria")
    parser.add_argument('--edge_goal', action="store_true",
                        help="if set, the agent will add the edge goal to navigate")
    
    # debugging
    parser.add_argument('--save_step_data', action="store_true", help='if true, save all debug data')
    parser.add_argument('--debug', action="store_true", help='if true, raise exceptions; if false, print exceptions and continue')
    parser.add_argument('--debug_frontier_score', action="store_false", help='label frontier in semantic map and print scores in visualization')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help="skip the first n frames")
    parser.add_argument('--keyboard_actor', action="store_true",
                        help="if set, the actor will be controlled by the keyboard")
    # parser.add_argument('--stair_climbing_agent', action="store_true")
    # parser.add_argument('--no_resume', action="store_true")
    # parser.add_argument('--stair_aim', type=int, default=1,
    #                     help="upstair=1, downstair=0")
    parser.add_argument('--no_stair_climbing', action="store_true",
                        help="if set, the agent will not climb stairs")
    parser.add_argument('--save_perception_results', action="store_true",
                        help="if set, the perception results will be saved")
    parser.add_argument("--gt_perception", action="store_true", help="if set, the gt perception will be generated")
    parser.add_argument("--gt_scenegraph", action="store_true",
                        help="if set, the gt scene graph will be generated")
    parser.add_argument("--new_mapping", action="store_true",
                        help="if set, the new mapping method will be used to try to mitigate incorrect occupancy")
    
    # Tunable Parameters
    parser.add_argument('--GRADIENT_THRESHOLD', type=float, default=0.3,
                        help='Threshold for height gradient to be considered as stairs (default: 0.3)')
    parser.add_argument('--OBSTACLE_HEIGHT', type=float, default=0.3,
                        help='Height threshold for obstacles (default: 0.3)')
    parser.add_argument('--LOCAL_GRID_RANGE', type=int, default=40,
                        help='Radius for local region to compute the height gradient (default: 40)')
    parser.add_argument('--FLOOR_HEIGHT', type=float, default=1.0,
                        help='Height threshold for different floors (default: 1.0)')
    parser.add_argument('--FLOOR_POINT_THRESHOLD', type=int, default=800,
                        help='Threshold for new floor detection (default: 800)')
    parser.add_argument('--ACTUAL_CAMERA_HEIGHT', type=float, default=1.0,
                        help='Actual camera height (default: 1.0)')
    parser.add_argument('--AT_FRONTEIR_THRESHOLD', type=int, default=6,
                        help='Threshold for the robot to be considered at the goal (default: 6)')
    parser.add_argument('--NEAR_GOAL_DISTANCE', type=float, default=8,
                        help='Threshold for the robot to be considered at the goal (default: 8)')
    # parse arguments
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    return args
