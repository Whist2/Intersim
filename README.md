首先生成hdmap视频，建议在本地用intersim项目（因为有gui），先配置环境：
conda create -n intersim python=3.10
pip install -r requirements.txt --no-deps#因为有一些项pip会认为冲突，但是requirements中的包是全的，用--no-deps可以避免报错
conda install pyqt#需要另外安装pyqt否则也会出问题
python simple_usage_example.py
之后先选场景，再在左侧静态图中选中车，退出gui，即可生成对应视角的视频
视频保存在outputs\example\rds_hq\{编号}\render\hdmap

cosmos渲染：
在超算平台里输入：
cd /share/home/u22537/data/DXW/cosmos-transfer1
srun -p A800 -n 1 --cpus-per-task=7 --gres=gpu:a800:1 --job-name=intersim --pty /bin/bash#至少需要40G显存，L40刚好差一些
source /share/apps/miniconda3/etc/profile.d/conda.sh#如果用sbatch需要这一步
conda init#如果用sbatch需要这一步
conda activate cosmos-transfer1
export HF_ENDPOINT=https://hf-mirror.com#中间会连huggingface，因此需要设置为镜像站，否则会报错
 ./render_hdmap.sh(单个视频）
 ./render_single2multiview_hdmap.sh（多个视频）
 
在render_hdmap.sh中，export HDMAP_VIDEO后为输入的图片地址
export PROMPT为渲染该图片的prompt
运行完毕后输出到OUTPUT_FOLDER对应的地址，修改--video_save_name参数以修改输出视频的名称

在render_single2multiview_hdmap中，HDMAP_BASE_DIR设置六个视角视频文件夹的上级文件夹（即HDMAP_BASE_DIR-六个视角文件夹-六个视角视频）
# Camera-specific prompts 后可以设置6个视角的prompt
