#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AVA Model Pipeline - 自动打包脚本
使用 PyInstaller 将 1-pipe.py 打包为 AutoAdmix_v1.exe
"""

import os
import sys
import stat
import shutil
import subprocess
from pathlib import Path

# 处理 Windows 控制台编码问题
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # 设置控制台编码为 UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


class Packager:
    def __init__(self):
        self.pipe_dir = Path(__file__).parent
        self.build_dir = self.pipe_dir / "build"
        self.dist_dir = self.pipe_dir / "dist"
        self.spec_file = self.pipe_dir / "AVA-Pipeline.spec"
        self.exe_file = self.dist_dir / "AutoAdmix_v1.exe"
        
    def print_header(self, text):
        """打印标题"""
        print("\n" + "━" * 60)
        print(f"  {text}")
        print("━" * 60 + "\n")
        
    def print_step(self, step_num, total, text):
        """打印步骤"""
        print(f"[{step_num}/{total}] {text}...")
        
    def print_success(self, text):
        """打印成功信息"""
        print(f"✓ {text}")
        
    def print_error(self, text):
        """打印错误信息"""
        print(f"❌ 错误: {text}", file=sys.stderr)
        
    def print_warning(self, text):
        """打印警告信息"""
        print(f"⚠ 警告: {text}")
    
    def check_files(self):
        """检查必要文件"""
        self.print_step(1, 4, "检查必要文件")
        
        required_files = {
            "1-pipe.py": "主程序文件",
            "AVA-Pipeline.spec": "打包配置文件",
        }
        
        for filename, desc in required_files.items():
            filepath = self.pipe_dir / filename
            if not filepath.exists():
                self.print_error(f"找不到 {desc}: {filename}")
                return False
            self.print_success(f"找到 {desc}: {filename}")
        
        return True
    
    def check_pyinstaller(self):
        """检查 PyInstaller"""
        self.print_step(2, 4, "检查 PyInstaller")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "PyInstaller", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_success(f"PyInstaller {version}")
                return True
            else:
                self.print_error("PyInstaller 版本检查失败")
                return False
        except Exception as e:
            self.print_error(f"无法检查 PyInstaller: {e}")
            return False
    
    def clean_old_builds(self):
        """清理旧的编译结果"""
        self.print_step(3, 4, "清理旧的编译结果")
        
        def handle_remove_error(func, path, exc):
            """处理删除错误"""
            try:
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                    func(path)
                else:
                    func(path)
            except Exception:
                pass  # 忽略无法删除的文件
        
        # 删除 build 目录
        if self.build_dir.exists():
            try:
                shutil.rmtree(self.build_dir, onerror=handle_remove_error)
                self.print_success("删除 build 目录")
            except Exception as e:
                self.print_warning(f"清理 build 目录失败: {e}")
        
        # 删除旧的 exe 文件
        if self.exe_file.exists():
            try:
                os.chmod(str(self.exe_file), stat.S_IWUSR | stat.S_IRUSR)
                self.exe_file.unlink()
                self.print_success("删除旧的 exe 文件")
            except Exception as e:
                self.print_warning(f"删除旧 exe 失败: {e}")
        
        # 确保 dist 目录存在
        if not self.dist_dir.exists():
            self.dist_dir.mkdir(parents=True, exist_ok=True)
            self.print_success("创建 dist 目录")
        else:
            self.print_success("dist 目录已存在")
    
    def run_pyinstaller(self):
        """运行 PyInstaller"""
        self.print_step(4, 4, "运行 PyInstaller")
        
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--distpath", str(self.dist_dir),
            str(self.spec_file)
        ]
        
        print(f"\n执行命令: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, cwd=str(self.pipe_dir))
            return result.returncode == 0
        except Exception as e:
            self.print_error(f"PyInstaller 执行失败: {e}")
            return False
    
    def verify_output(self):
        """验证输出"""
        if not self.exe_file.exists():
            self.print_error("找不到生成的 exe 文件")
            return False
        
        file_size_mb = self.exe_file.stat().st_size / (1024 * 1024)
        
        self.print_header("✓ 打包成功！")
        print(f"输出文件: {self.exe_file}")
        print(f"文件大小: {file_size_mb:.2f} MB")
        print(f"\n您可以运行以下命令启动应用:")
        print(f"  {self.exe_file}")
        
        return True
    
    def pack(self):
        """执行打包"""
        self.print_header("AVA Model Pipeline - 自动打包脚本")
        
        if not self.check_files():
            return False
        
        if not self.check_pyinstaller():
            self.print_warning("PyInstaller 可能未安装或无法访问")
            # 继续尝试，可能会失败
        
        self.clean_old_builds()
        
        print("\n" + "━" * 60)
        print("  开始打包...")
        print("━" * 60)
        
        if not self.run_pyinstaller():
            self.print_error("打包失败")
            return False
        
        if not self.verify_output():
            return False
        
        return True


def main():
    """主函数"""
    packager = Packager()
    
    try:
        success = packager.pack()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n打包被中止")
        sys.exit(130)
    except Exception as e:
        packager.print_error(f"未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
