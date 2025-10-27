#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 交互式标注GUI工具 - 多bbox版本
按键说明：
- 鼠标左键：添加正样本点（红色）
- 鼠标右键：添加负样本点（蓝色）
- R键：重置当前图片的所有标注
- 右箭头：保存并前进到下一张
- 左箭头：返回上一张重新标注
- ESC：退出程序
"""

import os
import json
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import Tk, Label, Canvas, Frame, Button
from pathlib import Path
import pickle
import cv2

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
except ImportError as e:
    print("错误: 请先安装 SAM2")
    print("参考: https://github.com/facebookresearch/segment-anything-2")
    print(f"详细错误: {e}")
    exit(1)


class SAM2InteractiveGUI:
    def __init__(self,
                 input_folder="vggss_test_multi",
                 json_path="vggss.json",
                 output_dir="vggss_seg_manual_multi",
                 sam2_checkpoint="./checkpoints_sam/sam2.1_hiera_large.pt",
                 model_cfg="sam2.1/sam2.1_hiera_l",
                 device='cuda'):

        self.root = Tk()
        self.root.title("SAM2 交互式标注工具 - 多BBox版本")

        # 设置文件夹路径
        self.input_folder = Path(input_folder)
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 创建输出文件夹
        self.combined_dir = self.output_dir / 'combined'
        self.masks_dir = self.output_dir / 'masks'
        self.combined_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        # 加载SAM2模型
        print("加载SAM2模型...")
        self.load_sam2(sam2_checkpoint, model_cfg)

        # 加载数据
        print("加载数据...")
        self.load_data()

        if not self.images_to_process:
            print("错误：没有需要处理的图片！")
            self.root.destroy()
            return

        print(f"找到 {len(self.images_to_process)} 张图片需要标注")

        # 当前状态
        self.current_index = 0
        self.history = []

        # 当前图片的标注数据
        self.current_image = None
        self.current_image_np = None
        self.current_mask = None  # 当前正在标注的物体mask
        self.accumulated_mask = None  # 累积的所有已确认物体mask
        self.current_bboxes = []  # 当前图片的所有bbox列表
        self.positive_points = []  # [(x, y), ...]
        self.negative_points = []  # [(x, y), ...]
        self.photo = None  # 保存PhotoImage引用
        self.confirmed_objects_count = 0  # 已确认的物体数量

        # 显示设置
        self.display_size = 512  # 显示尺寸
        self.scale_factor = 1.0  # 缩放因子

        # 创建UI
        self.setup_ui()

        # 绑定事件
        self.canvas.bind('<Button-1>', self.on_left_click)  # 左键：正样本
        self.canvas.bind('<Button-2>', self.on_right_click)  # 右键：负样本 (macOS)
        self.canvas.bind('<Button-3>', self.on_right_click)  # 右键：负样本 (Windows/Linux)
        self.root.bind('r', self.reset_annotation)
        self.root.bind('R', self.reset_annotation)
        self.root.bind('<space>', self.confirm_current_object)  # 空格：确认当前物体
        self.root.bind('c', self.clear_accumulated_masks)
        self.root.bind('C', self.clear_accumulated_masks)
        self.root.bind('<Left>', self.go_previous)
        self.root.bind('<Right>', self.go_next)
        self.root.bind('<Escape>', lambda e: self.root.quit())

        # 显示第一张图片
        self.display_image()

    def load_sam2(self, sam2_checkpoint, model_cfg):
        """加载SAM2模型"""
        # 获取配置文件路径
        config_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "segment-anything-2/sam2/configs"
        ))
        config_name = model_cfg.replace('.yaml', '')

        print(f"配置目录: {config_dir}")
        print(f"配置名称: {config_name}")

        # 清除之前的 Hydra 初始化
        GlobalHydra.instance().clear()

        # 使用绝对路径初始化 Hydra
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            cfg = compose(config_name=config_name)
            OmegaConf.resolve(cfg)
            sam2_model = instantiate(cfg.model, _recursive_=True)

        # 加载权重
        if sam2_checkpoint and os.path.exists(sam2_checkpoint):
            print(f"加载权重: {sam2_checkpoint}")
            state_dict = torch.load(sam2_checkpoint, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            sam2_model.load_state_dict(state_dict, strict=True)

        sam2_model = sam2_model.to(self.device)
        sam2_model.eval()

        self.predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2模型加载完成！")

    def load_data(self):
        """加载图片列表和JSON数据"""
        # 读取input文件夹
        input_files = []
        for file in self.input_folder.iterdir():
            if file.suffix.lower() == '.jpg':
                input_files.append(file.name)

        print(f"找到 {len(input_files)} 张图片")

        # 读取JSON
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # 建立映射 - 修改：保存所有bbox
        file_info_map = {}
        for item in data:
            filename = item['file'] + '.jpg'
            file_info_map[filename] = {
                'label': item['class'],
                'bboxes': item['bbox'] if item['bbox'] else []  # 保存所有bbox
            }

        # 构建待处理图片列表（跳过已标注的）
        self.images_to_process = []
        skipped_count = 0

        for filename in sorted(input_files):
            if filename in file_info_map:
                base_name = os.path.splitext(filename)[0]

                # 检查是否已经标注过（检查combined图片和mask pkl）
                combined_exists = (self.combined_dir / filename).exists()
                mask_exists = (self.masks_dir / f"{base_name}.pkl").exists()

                if combined_exists or mask_exists:
                    # 已经标注过，跳过
                    skipped_count += 1
                    continue

                # 未标注，加入待处理列表
                self.images_to_process.append({
                    'filename': filename,
                    'base_name': base_name,
                    'label': file_info_map[filename]['label'],
                    'bboxes': file_info_map[filename]['bboxes']  # 保存所有bbox
                })

        print(f"已标注图片: {skipped_count} 张")
        print(f"待标注图片: {len(self.images_to_process)} 张")

    def setup_ui(self):
        """设置UI界面"""
        # 信息标签
        self.info_label = Label(
            self.root,
            text="",
            font=("Arial", 14, "bold"),
            pady=10
        )
        self.info_label.pack()

        # Label标签
        self.label_info = Label(
            self.root,
            text="",
            font=("Arial", 12),
            fg="blue",
            pady=5
        )
        self.label_info.pack()

        # 图片尺寸信息
        self.size_info = Label(
            self.root,
            text="",
            font=("Arial", 10),
            fg="gray",
            pady=2
        )
        self.size_info.pack()

        # BBox数量信息
        self.bbox_count_info = Label(
            self.root,
            text="",
            font=("Arial", 10),
            fg="purple",
            pady=2
        )
        self.bbox_count_info.pack()

        # 物体计数信息
        self.object_count_info = Label(
            self.root,
            text="已确认物体: 0",
            font=("Arial", 11, "bold"),
            fg="green",
            pady=5
        )
        self.object_count_info.pack()

        # 按钮区域
        button_frame = Frame(self.root)
        button_frame.pack(pady=5)

        self.reset_btn = Button(
            button_frame,
            text="Reset (R)",
            command=lambda: self.reset_annotation(None),
            font=("Arial", 11),
            bg='orange',
            fg='white',
            padx=20,
            pady=5
        )
        self.reset_btn.pack()

        # 画布
        self.canvas = Canvas(
            self.root,
            width=self.display_size,
            height=self.display_size,
            bg='gray',
            cursor='crosshair'
        )
        self.canvas.pack(padx=10, pady=10)

        # 帮助信息
        help_text = (
            "操作说明：\n"
            "鼠标左键：添加正样本点（红色，包括区域）\n"
            "鼠标右键：添加负样本点（蓝色，排除区域）\n"
            "空格键：确认当前物体，开始标注下一个物体\n"
            "R键 / Reset按钮：重置当前物体的标注点\n"
            "C键：清除所有累积的mask，重新开始\n"
            "→ 右箭头：保存并前进到下一张\n"
            "← 左箭头：返回上一张重新标注\n"
            "ESC：退出程序\n\n"
            "颜色说明：\n"
            "深绿色 = 已确认的物体\n"
            "亮绿色 = 当前正在标注的物体\n"
            "红色框 = 标注的所有BBox（多个）"
        )
        self.help_label = Label(
            self.root,
            text=help_text,
            font=("Arial", 10),
            justify="left",
            pady=10
        )
        self.help_label.pack()

    def display_image(self):
        """显示当前图片"""
        if self.current_index >= len(self.images_to_process):
            self.show_completion()
            return

        # 获取当前图片信息
        current_item = self.images_to_process[self.current_index]
        filename = current_item['filename']
        label = current_item['label']
        self.current_bboxes = current_item['bboxes']  # 保存所有bbox

        # 更新信息标签
        info_text = f"图片 {self.current_index + 1} / {len(self.images_to_process)}: {filename}"
        self.info_label.config(text=info_text)
        self.label_info.config(text=f"类别: {label}")
        self.bbox_count_info.config(text=f"BBox数量: {len(self.current_bboxes)}")

        print(f"\n正在显示: {filename}")
        print(f"BBox数量: {len(self.current_bboxes)}")

        # 加载图片
        image_path = self.input_folder / filename
        self.current_image = Image.open(image_path).convert('RGB')
        self.current_image_np = np.array(self.current_image)

        # 计算缩放因子
        orig_w, orig_h = self.current_image.size
        self.scale_factor = self.display_size / max(orig_w, orig_h)
        new_w = int(orig_w * self.scale_factor)
        new_h = int(orig_h * self.scale_factor)

        # 更新尺寸信息
        self.size_info.config(text=f"原图尺寸: {orig_w} x {orig_h}")

        # 设置图片到SAM2
        self.predictor.set_image(self.current_image_np)

        # 重置标注（切换图片时清空所有mask和点）
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.accumulated_mask = None
        self.confirmed_objects_count = 0

        # 更新物体计数显示
        self.object_count_info.config(text="已确认物体: 0")

        # 显示图片
        self.update_display()

    def update_display(self):
        """更新显示（包括图片、mask、点、所有bbox）"""
        if self.current_image is None:
            print("警告: current_image 为空")
            return

        try:
            # 创建显示图片
            display_img = self.current_image.copy()
            img_np = np.array(display_img)

            # 如果有accumulated_mask或current_mask，叠加显示
            if self.accumulated_mask is not None or self.current_mask is not None:
                overlay = img_np.copy()

                # 1. 先绘制已确认的accumulated_mask（深绿色）
                if self.accumulated_mask is not None:
                    overlay[self.accumulated_mask] = overlay[self.accumulated_mask] * 0.3 + np.array([0, 180, 0]) * 0.7

                    # 绘制accumulated mask的轮廓（深黄色）
                    contours, _ = cv2.findContours(
                        self.accumulated_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    overlay = cv2.drawContours(overlay, contours, -1, (200, 200, 0), 2)

                # 2. 再绘制当前正在标注的current_mask（亮绿色半透明）
                if self.current_mask is not None:
                    overlay[self.current_mask] = overlay[self.current_mask] * 0.5 + np.array([0, 255, 0]) * 0.5

                    # 绘制current mask的轮廓（黄色）
                    contours, _ = cv2.findContours(
                        self.current_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    overlay = cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

                display_img = Image.fromarray(overlay.astype(np.uint8))

            # 验证图片有效性
            if display_img is None or display_img.size[0] == 0 or display_img.size[1] == 0:
                print("警告: 图片无效或尺寸为0")
                return

            # 缩放图片
            orig_w, orig_h = display_img.size
            new_w = int(orig_w * self.scale_factor)
            new_h = int(orig_h * self.scale_factor)

            # 确保缩放后尺寸有效
            if new_w <= 0 or new_h <= 0:
                print(f"警告: 缩放后尺寸无效 {new_w}x{new_h}")
                return

            display_img = display_img.resize((new_w, new_h), Image.LANCZOS)

            # 绘制bbox和点
            draw = ImageDraw.Draw(display_img)

            # 绘制所有bbox参考框（红色虚线效果） - 修改：循环绘制所有bbox
            bbox_colors = ['red', 'orange', 'yellow', 'pink', 'cyan']  # 多种颜色区分不同bbox
            for idx, bbox in enumerate(self.current_bboxes):
                try:
                    bbox_x1 = int(bbox[0] * orig_w * self.scale_factor)
                    bbox_y1 = int(bbox[1] * orig_h * self.scale_factor)
                    bbox_x2 = int(bbox[2] * orig_w * self.scale_factor)
                    bbox_y2 = int(bbox[3] * orig_h * self.scale_factor)

                    # 验证bbox坐标是否有效
                    if bbox_x2 > bbox_x1 and bbox_y2 > bbox_y1:
                        # 选择颜色
                        color = bbox_colors[idx % len(bbox_colors)]

                        # 绘制彩色半透明矩形框
                        draw.rectangle(
                            [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                            outline=color,
                            width=3
                        )

                        # 在bbox左上角添加"BBox N"标签
                        try:
                            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
                        except:
                            try:
                                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                            except:
                                font = ImageFont.load_default()

                        # 绘制标签背景
                        text = f"BBox {idx+1}"
                        text_bbox = draw.textbbox((bbox_x1, bbox_y1 - 18), text, font=font)
                        draw.rectangle(text_bbox, fill=color)
                        draw.text((bbox_x1, bbox_y1 - 18), text, fill='white', font=font)
                    else:
                        print(f"警告: bbox {idx+1} 坐标无效，不显示bbox框 - [{bbox_x1}, {bbox_y1}, {bbox_x2}, {bbox_y2}]")
                except Exception as e:
                    print(f"警告: bbox {idx+1} 绘制出错，不显示bbox框 - {e}")

            point_radius = 5

            # 绘制正样本点（红色）
            for px, py in self.positive_points:
                x = int(px * self.scale_factor)
                y = int(py * self.scale_factor)
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill='red',
                    outline='white',
                    width=2
                )

            # 绘制负样本点（蓝色）
            for px, py in self.negative_points:
                x = int(px * self.scale_factor)
                y = int(py * self.scale_factor)
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill='blue',
                    outline='white',
                    width=2
                )

            # 显示在画布上
            # 先清空画布
            self.canvas.delete("all")

            # 强制更新，确保删除操作完成
            self.canvas.update_idletasks()

            # 删除旧的PhotoImage引用
            if hasattr(self, 'photo'):
                del self.photo

            # 创建PhotoImage（必须保持引用防止被垃圾回收）
            self.photo = ImageTk.PhotoImage(display_img)

            # 居中显示
            x_offset = (self.display_size - new_w) // 2
            y_offset = (self.display_size - new_h) // 2

            # 创建图片对象
            self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo)

            # 强制更新画布
            self.canvas.update()

            # 保存偏移量，用于点击坐标转换
            self.display_offset = (x_offset, y_offset)
            self.display_dims = (new_w, new_h)

        except Exception as e:
            print(f"更新显示时出错: {e}")
            import traceback
            traceback.print_exc()
            # 清空画布显示错误
            self.canvas.delete("all")
            self.canvas.create_text(
                self.display_size // 2,
                self.display_size // 2,
                text=f"显示图片出错\n{str(e)}",
                fill='red',
                font=("Arial", 12)
            )

    def on_left_click(self, event):
        """鼠标左键点击：添加正样本点"""
        x, y = self.canvas_to_image_coords(event.x, event.y)
        if x is not None:
            self.positive_points.append((x, y))
            print(f"添加正样本点: ({x}, {y})")
            self.update_prediction()

    def on_right_click(self, event):
        """鼠标右键点击：添加负样本点"""
        x, y = self.canvas_to_image_coords(event.x, event.y)
        if x is not None:
            self.negative_points.append((x, y))
            print(f"添加负样本点: ({x}, {y})")
            self.update_prediction()

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """将画布坐标转换为图片坐标"""
        if not hasattr(self, 'display_offset') or not hasattr(self, 'display_dims'):
            return None, None

        x_offset, y_offset = self.display_offset
        disp_w, disp_h = self.display_dims

        # 转换为显示图片坐标
        x = canvas_x - x_offset
        y = canvas_y - y_offset

        # 检查是否在图片范围内
        if x < 0 or x >= disp_w or y < 0 or y >= disp_h:
            return None, None

        # 转换为原图坐标
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)

        return orig_x, orig_y

    def update_prediction(self):
        """更新SAM2预测"""
        if not self.positive_points and not self.negative_points:
            self.current_mask = None
            self.update_display()
            return

        # 准备point_coords和point_labels
        point_coords = []
        point_labels = []

        for px, py in self.positive_points:
            point_coords.append([px, py])
            point_labels.append(1)  # 正样本

        for px, py in self.negative_points:
            point_coords.append([px, py])
            point_labels.append(0)  # 负样本

        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)

        # 使用SAM2预测
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

            self.current_mask = masks[0].astype(bool)
            print(f"预测完成，mask尺寸: {self.current_mask.shape}")

        except Exception as e:
            print(f"预测出错: {e}")
            import traceback
            traceback.print_exc()

        # 更新显示
        self.update_display()

    def reset_annotation(self, event=None):
        """重置当前物体的标注（不清除已确认的物体）"""
        print("重置当前物体标注")
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None
        self.update_display()

    def confirm_current_object(self, event=None):
        """确认当前物体，将mask累积并清空点开始标注下一个物体"""
        if self.current_mask is None:
            print("没有当前mask，无法确认")
            return

        # 将当前mask累积到accumulated_mask
        if self.accumulated_mask is None:
            self.accumulated_mask = self.current_mask.copy()
        else:
            # 合并mask（使用逻辑OR）
            self.accumulated_mask = np.logical_or(self.accumulated_mask, self.current_mask)

        self.confirmed_objects_count += 1
        print(f"✓ 已确认物体 {self.confirmed_objects_count}，可以继续标注下一个物体")

        # 更新物体计数显示
        self.object_count_info.config(text=f"已确认物体: {self.confirmed_objects_count}")

        # 清空当前标注，准备标注下一个物体
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None

        # 更新显示
        self.update_display()

    def clear_accumulated_masks(self, event=None):
        """清除所有累积的mask，重新开始"""
        print("清除所有累积的mask")
        self.accumulated_mask = None
        self.current_mask = None
        self.positive_points = []
        self.negative_points = []
        self.confirmed_objects_count = 0

        # 更新物体计数显示
        self.object_count_info.config(text="已确认物体: 0")

        self.update_display()

    def save_current_annotation(self):
        """保存当前标注（合并accumulated_mask和current_mask）"""
        # 检查是否有任何mask需要保存
        if self.current_mask is None and self.accumulated_mask is None:
            print("当前没有标注，跳过保存")
            return False

        current_item = self.images_to_process[self.current_index]
        filename = current_item['filename']
        base_name = current_item['base_name']
        label = current_item['label']
        bboxes = current_item['bboxes']  # 所有bbox

        try:
            # 合并accumulated_mask和current_mask
            if self.accumulated_mask is not None and self.current_mask is not None:
                # 两者都存在，合并
                final_mask = np.logical_or(self.accumulated_mask, self.current_mask)
            elif self.accumulated_mask is not None:
                # 只有accumulated_mask
                final_mask = self.accumulated_mask
            else:
                # 只有current_mask
                final_mask = self.current_mask

            # 1. 保存mask为pkl
            mask_pkl_path = self.masks_dir / f"{base_name}.pkl"
            original_size = self.current_image.size
            self.save_mask_pkl(final_mask, original_size, mask_pkl_path)

            # 2. 生成combined图片（使用合并后的mask和所有bbox）
            self.generate_combined_image(filename, base_name, label, bboxes, final_mask)

            print(f"✓ 已保存标注: {filename}")
            if self.confirmed_objects_count > 0:
                print(f"  包含 {self.confirmed_objects_count} 个已确认物体" +
                      (" + 1个当前物体" if self.current_mask is not None else ""))
            return True

        except Exception as e:
            print(f"保存标注时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_mask_pkl(self, mask, original_size, output_path):
        """保存mask为pkl文件"""
        original_width, original_height = original_size

        # 确保mask尺寸与原图一致
        if mask.shape != (original_height, original_width):
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_image = mask_image.resize((original_width, original_height), Image.NEAREST)
            mask = np.array(mask_image) > 0

        # 转换为 (height, width, 1) 的 float64 格式
        mask_array = mask.astype(np.float64)[:, :, np.newaxis]

        # 保存为pkl文件
        with open(output_path, 'wb') as f:
            pickle.dump(mask_array, f)

    def generate_combined_image(self, filename, base_name, label, bboxes, mask):
        """生成combined图片（原图 | bbox | mask）- 修改：绘制所有bbox"""
        # 1. 原图
        original = self.current_image.copy()

        # 2. bbox图 - 修改：绘制所有bbox
        bbox_image = self.current_image.copy()
        if bboxes:
            try:
                draw_bbox = ImageDraw.Draw(bbox_image)
                w, h = bbox_image.size

                # 绘制所有bbox
                bbox_colors = ['red', 'orange', 'yellow', 'pink', 'cyan']
                for idx, bbox in enumerate(bboxes):
                    bbox_pixel = [
                        int(bbox[0] * w),
                        int(bbox[1] * h),
                        int(bbox[2] * w),
                        int(bbox[3] * h)
                    ]

                    # 验证bbox坐标有效性
                    if bbox_pixel[2] > bbox_pixel[0] and bbox_pixel[3] > bbox_pixel[1]:
                        color = bbox_colors[idx % len(bbox_colors)]
                        draw_bbox.rectangle(bbox_pixel, outline=color, width=3)

                        # 添加标签（只在第一个bbox上添加类别名）
                        if idx == 0:
                            try:
                                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                            except:
                                try:
                                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                                except:
                                    font = ImageFont.load_default()

                            text_bbox = draw_bbox.textbbox((bbox_pixel[0], bbox_pixel[1] - 20), label, font=font)
                            draw_bbox.rectangle(text_bbox, fill=color)
                            draw_bbox.text((bbox_pixel[0], bbox_pixel[1] - 20), label, fill='white', font=font)
                    else:
                        print(f"  注意: bbox {idx+1} 坐标无效，保存的combined图中不显示该bbox")
            except Exception as e:
                print(f"  注意: bbox处理出错，保存的combined图中不显示bbox - {e}")

        # 3. mask overlay图
        img_np = np.array(self.current_image)
        overlay = img_np.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

        # 绘制轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        overlay = cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

        mask_image = Image.fromarray(overlay.astype(np.uint8))

        # 4. 调整所有图片到224x224
        original = original.resize((224, 224), Image.LANCZOS)
        bbox_image = bbox_image.resize((224, 224), Image.LANCZOS)
        mask_image = mask_image.resize((224, 224), Image.LANCZOS)

        # 5. 拼接成672x224
        combined = Image.new('RGB', (672, 224))
        combined.paste(original, (0, 0))
        combined.paste(bbox_image, (224, 0))
        combined.paste(mask_image, (448, 0))

        # 添加标题
        draw = ImageDraw.Draw(combined)
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                title_font = ImageFont.load_default()

        draw.text((5, 5), "Original", fill='white', font=title_font)
        draw.text((229, 5), f"BBox ({len(bboxes)})", fill='white', font=title_font)
        draw.text((453, 5), "Mask", fill='white', font=title_font)

        # 保存
        combined_output = self.combined_dir / filename
        combined.save(combined_output)

    def go_next(self, event=None):
        """前进到下一张"""
        # 如果有标注，先保存
        if self.current_mask is not None or self.accumulated_mask is not None:
            self.save_current_annotation()

        if self.current_index < len(self.images_to_process):
            self.history.append(self.current_index)
            self.current_index += 1
            self.display_image()

    def go_previous(self, event=None):
        """返回上一张"""
        if self.history:
            self.current_index = self.history.pop()
            self.display_image()
        else:
            print("已经是第一张图片了")

    def show_completion(self):
        """显示完成消息"""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.display_size // 2,
            self.display_size // 2,
            text="所有图片已处理完成！\n按 ESC 退出",
            font=("Arial", 20, "bold"),
            fill="white"
        )
        self.info_label.config(text="完成！")

    def run(self):
        """运行GUI"""
        if self.images_to_process:
            self.root.mainloop()


if __name__ == "__main__":
    # ============ 配置 ============
    INPUT_FOLDER = 'vggss_test_multi'
    JSON_PATH = 'vggss.json'
    OUTPUT_DIR = 'vggss_seg_manual_multi'

    # SAM2模型配置
    SAM2_CHECKPOINT = "./checkpoints_sam/sam2.1_hiera_large.pt"
    MODEL_CFG = "sam2.1/sam2.1_hiera_l"

    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============ 运行 ============
    print("SAM2 交互式标注工具 - 多BBox版本")
    print("=" * 60)
    print(f"输入文件夹: {INPUT_FOLDER}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"设备: {DEVICE}")
    print("=" * 60)
    print()

    app = SAM2InteractiveGUI(
        input_folder=INPUT_FOLDER,
        json_path=JSON_PATH,
        output_dir=OUTPUT_DIR,
        sam2_checkpoint=SAM2_CHECKPOINT,
        model_cfg=MODEL_CFG,
        device=DEVICE
    )
    app.run()

    print("\n完成！")
