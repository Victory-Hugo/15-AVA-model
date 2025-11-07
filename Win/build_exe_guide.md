# 构建 AVA Pipeline GUI 的步骤

1. **准备虚拟环境并安装依赖**
   ```powershell
   cd C:\Users\Administrator\Desktop
   python -m venv .venv_ava
   .\.venv_ava\Scripts\activate
   pip install --upgrade pip
   pip install PyQt5 pyinstaller numpy pandas scipy scikit-learn tabulate
   ```

2. **执行 PyInstaller 打包命令**
   ```powershell
   .\.venv_ava\Scripts\pyinstaller --noconfirm --onefile --windowed `
     --name AVA_Pipeline_GUI `
     --add-data "F:\OneDrive\文档（科研）\脚本\Download\15-AVA-model\Win\python;python" `
     --add-data "F:\OneDrive\文档（科研）\脚本\Download\15-AVA-model\Win\data;DATA" `
     ava_pipeline_gui.py
   ```

生成的可执行文件位于 `C:\Users\Administrator\Desktop\dist\AVA_Pipeline_GUI.exe`，将整个 `dist` 目录保留即可直接分发。
