



def parse_args():

    parser = argparse.ArgumentParser()
    
    # CLI args
    
    parser.add_argument('--train_batch_size', 
                        type=int, 
                        default=64)
    
    parser.add_argument('--train_steps_per_epoch',
                        type=int,
                        default=64)

    parser.add_argument('--validation_batch_size', 
                        type=int, 
                        default=64)
    
    parser.add_argument('--validation_steps_per_epoch',
                        type=int,
                        default=64)

    parser.add_argument('--epochs', 
                        type=int, 
                        default=1)
    
    parser.add_argument('--freeze_bert_layer', 
                        type=eval, 
                        default=False)
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.01)
    
    parser.add_argument('--momentum', 
                        type=float, 
                        default=0.5)
    
    parser.add_argument('--seed', 
                        type=int, 
                        default=42)
    
    parser.add_argument('--log_interval', 
                        type=int, 
                        default=100)
    
    parser.add_argument('--backend', 
                        type=str, 
                        default=None)
    
    parser.add_argument('--max_seq_length', 
                        type=int, 
                        default=128)
    
    parser.add_argument('--run_validation', 
                        type=eval,
                        default=False)
        
    # Container environment  
    
    parser.add_argument('--hosts', 
                        type=list, 
                        default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--current_host', 
                        type=str, 
                        default=os.environ['SM_CURRENT_HOST'])
    
    parser.add_argument('--model_dir', 
                        type=str, 
                        default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--train_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--validation_data', 
                        type=str, 
                        default=os.environ['SM_CHANNEL_VALIDATION'])
        
    parser.add_argument('--output_dir', 
                        type=str, 
                        default=os.environ['SM_OUTPUT_DIR'])
    
    parser.add_argument('--num_gpus', 
                        type=int, 
                        default=os.environ['SM_NUM_GPUS'])

    # Debugger args
    
    parser.add_argument("--save-frequency", 
                        type=int, 
                        default=10, 
                        help="frequency with which to save steps")
    
    parser.add_argument("--smdebug_path",
                        type=str,
                        help="output directory to save data in",
                        default="/opt/ml/output/tensors",)
    
    parser.add_argument("--hook-type",
                        type=str,
                        choices=["saveall", "module-input-output", "weights-bias-gradients"],
                        default="saveall",)

    return parser.parse_args()
    