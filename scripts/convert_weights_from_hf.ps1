# EDIT this line with your conda environment name
$CONDA_ENV_NAME="ttt-video"

$FINAL_SAVE_PATH="C:\Users\joshu\ttt-video-dit\base-model\converted"
$HUGGINGFACE_PRETRAINED_WEIGHTS_PATH="C:\Users\joshu\ttt-video-dit\base-model"
$SSM_TYPE="ttt_linear" # Either ttt_linear or ttt_mlp

# The directory existence check is now bypassed to allow overriding weights.
New-Item -ItemType Directory -Force -Path $FINAL_SAVE_PATH | Out-Null

& "C:\Users\joshu\AppData\Local\NVIDIA\MiniConda\condabin\conda.bat" run -n $CONDA_ENV_NAME python -m ttt.models.cogvideo.weight_conversion.from_hf `
	--final_save_path $FINAL_SAVE_PATH `
	--ssm_type $SSM_TYPE `
	--pretrained_weights_dir $HUGGINGFACE_PRETRAINED_WEIGHTS_PATH

Read-Host -Prompt "Press Enter to exit" 