### 本地 HDMap 视频生成 (Intersim)

建议在本地使用 Intersim 项目来生成视频，因为它提供了图形用户界面（GUI）。

#### 步骤 1: 环境配置

首先，请配置好您的 Conda 环境。

1.  **创建 Conda 环境**:
    ```bash
    conda create -n intersim python=3.10
    ```

2.  **激活环境**:
    ```bash
    conda activate intersim
    ```

3.  **安装依赖包**:
    使用 `--no-deps` 标志可以避免 pip 报告不必要的冲突。
    ```bash
    pip install -r requirements.txt --no-deps
    ```

4.  **安装 PyQT**:
    需要另外安装 `pyqt` 以确保程序正常运行。
    ```bash
    conda install pyqt
    ```

#### 步骤 2: 生成视频

1.  **运行示例脚本**:
    ```bash
    python simple_usage_example.py
    ```

2.  **GUI 操作**:
    -   运行脚本后，首先选择一个场景。
    -   在左侧的静态图中选中您想要生成视频的车辆。
    -   退出 GUI，程序将自动生成对应视角的视频。

#### 步骤 3: 查看输出

生成的视频将保存在以下目录中，其中 `{编号}` 对应您所选场景的编号。
```
outputs/example/rds_hq/{编号}/render/hdmap
```

---

### 超算平台 Cosmos 渲染

在超算平台上使用 Cosmos 进行视频的渲染。

#### 步骤 1: 平台设置与环境激活

1.  **登录与资源申请**:
    进入工作目录，并使用 `srun` 申请计算资源。请注意，此任务至少需要 40G 显存。
    ```bash
    cd /share/home/u22537/data/DXW/cosmos-transfer1
    srun -p A800 -n 1 --cpus-per-task=7 --gres=gpu:a800:1 --job-name=intersim --pty /bin/bash
    ```

2.  **激活 Conda 环境**:
    如果您使用 `sbatch` 提交作业，需要执行 `source` 和 `conda init`。
    ```bash
    source /share/apps/miniconda3/etc/profile.d/conda.sh
    conda init
    conda activate cosmos-transfer1
    ```

3.  **设置 Hugging Face 镜像**:
    为避免连接问题，需要将 Hugging Face 的端点设置为镜像站点。
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```

#### 步骤 2: 执行渲染脚本

您可以选择渲染单个或多个视频。

##### 渲染单个视频 (`render_hdmap.sh`)

-   **运行脚本**:
    ```bash
    ./render_hdmap.sh
    ```
-   **参数配置**:
    在 `render_hdmap.sh` 脚本内部，您需要配置以下参数：
    -   `HDMAP_VIDEO`: 设置输入的 `hdmap` 视频文件路径。
    -   `PROMPT`: 为该视频设置渲染用的文本提示（Prompt）。
    -   `OUTPUT_FOLDER`: 指定渲染后视频的输出目录。
    -   `--video_save_name`: 修改此参数以指定输出视频的文件名。

##### 渲染多个视频 (`render_single2multiview_hdmap.sh`)

-   **运行脚本**:
    ```bash
    ./render_single2multiview_hdmap.sh
    ```
-   **参数配置**:
    在 `render_single2multiview_hdmap.sh` 脚本内部，您需要配置以下参数：
    -   `HDMAP_BASE_DIR`: 设置包含六个不同视角视频文件夹的上级目录。其目录结构应为：`HDMAP_BASE_DIR -> 六个视角文件夹 -> 六个视角视频`。
    -   **Camera-specific prompts**: 在脚本中标有 `# Camera-specific prompts` 的部分之后，您可以为六个不同的摄像机视角分别设置独立的渲染提示。
