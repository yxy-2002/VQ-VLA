"""
可视化真机轨迹数据：主视角 + 腕部视角 + 完整12维动作 + 24维状态信息，合成为视频。

用法:
    python scripts/visualize_trajectory.py --traj_id 0
    python scripts/visualize_trajectory.py --traj_id 0 1 2        # 多条轨迹
    python scripts/visualize_trajectory.py --all                   # 所有成功轨迹
    python scripts/visualize_trajectory.py --traj_id 0 --fps 10
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


DATA_DIR = Path("/home/admin01/yxy/VQ-VLA/data/20260327-11:10:43/demos")
OUTPUT_DIR = Path("/home/admin01/yxy/VQ-VLA/visualizations")


def put_text(img, text, pos, font_scale=0.38, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def render_frame(main_img, wrist_img, action, state, step, total_steps, reward):
    """
    布局:
    +---------------------+---------------------+
    |   Main View         |   Wrist View        |
    |   (256x256)         |   (256x256)         |
    +---------------------+---------------------+
    |  Action (12-DoF)    |  State (24-dim)     |
    |  Arm 6D + Hand 6D  |  Full state info    |
    +---------------------+---------------------+
    """
    img_h, img_w = 256, 256
    text_area_h = 310
    canvas_w = img_w * 2
    canvas_h = img_h + text_area_h
    line_h = 16

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 放置图像
    canvas[0:img_h, 0:img_w] = cv2.resize(main_img, (img_w, img_h))
    canvas[0:img_h, img_w:img_w * 2] = cv2.resize(wrist_img, (img_w, img_h))

    # 图像标题
    put_text(canvas, "Main View", (5, 20), font_scale=0.55, color=(0, 255, 0), thickness=2)
    put_text(canvas, "Wrist View", (img_w + 5, 20), font_scale=0.55, color=(0, 255, 0), thickness=2)

    # 分隔线
    cv2.line(canvas, (0, img_h), (canvas_w, img_h), (100, 100, 100), 1)
    cv2.line(canvas, (img_w, 0), (img_w, canvas_h), (100, 100, 100), 1)

    # ==================== 左列: Step + Action ====================
    y = img_h + 8
    lx = 8  # 左列 x 起始

    # Step 信息
    put_text(canvas, f"Step: {step}/{total_steps - 1}   Reward: {reward:.1f}",
             (lx, y + line_h), font_scale=0.42, color=(0, 255, 255))

    # Arm action (dims 0-5)
    y_arm = y + line_h * 2 + 4
    put_text(canvas, "Action - Arm (6-DoF):", (lx, y_arm), font_scale=0.40, color=(180, 180, 255))
    put_text(canvas, f"  x  ={action[0]:+.4f}  y  ={action[1]:+.4f}  z  ={action[2]:+.4f}",
             (lx, y_arm + line_h), font_scale=0.35, color=(200, 200, 200))
    put_text(canvas, f"  roll={action[3]:+.4f}  pitch={action[4]:+.4f}  yaw={action[5]:+.4f}",
             (lx, y_arm + line_h * 2), font_scale=0.35, color=(200, 200, 200))

    # Hand action (dims 6-11)
    y_hand = y_arm + line_h * 3 + 6
    put_text(canvas, "Action - Hand (6-DoF):", (lx, y_hand), font_scale=0.40, color=(180, 180, 255))
    put_text(canvas, f"  d0={action[6]:+.4f}  d1={action[7]:+.4f}  d2={action[8]:+.4f}",
             (lx, y_hand + line_h), font_scale=0.35, color=(200, 200, 200))
    put_text(canvas, f"  d3={action[9]:+.4f}  d4={action[10]:+.4f}  d5={action[11]:+.4f}",
             (lx, y_hand + line_h * 2), font_scale=0.35, color=(200, 200, 200))

    # Action 数值条形图（直观展示12维动作的大小）
    y_bar = y_hand + line_h * 3 + 6
    put_text(canvas, "Action bar:", (lx, y_bar), font_scale=0.35, color=(150, 150, 150))
    bar_y0 = y_bar + 4
    bar_w_max = 100
    bar_h = 8
    labels_act = ["x", "y", "z", "R", "P", "Y", "h0", "h1", "h2", "h3", "h4", "h5"]
    for i in range(12):
        by = bar_y0 + i * (bar_h + 2)
        put_text(canvas, f"{labels_act[i]}", (lx, by + bar_h - 1), font_scale=0.28, color=(150, 150, 150))
        # 背景条
        cv2.rectangle(canvas, (lx + 22, by), (lx + 22 + bar_w_max, by + bar_h), (50, 50, 50), -1)
        # 数值条 (clamp to [-1, 1] for display)
        val = np.clip(action[i], -1.0, 1.0)
        center_x = lx + 22 + bar_w_max // 2
        bar_len = int(abs(val) * (bar_w_max // 2))
        color = (100, 200, 100) if i < 6 else (100, 150, 255)
        if val >= 0:
            cv2.rectangle(canvas, (center_x, by), (center_x + bar_len, by + bar_h), color, -1)
        else:
            cv2.rectangle(canvas, (center_x - bar_len, by), (center_x, by + bar_h), color, -1)
        # 中线
        cv2.line(canvas, (center_x, by), (center_x, by + bar_h), (200, 200, 200), 1)

    # ==================== 右列: State ====================
    rx = img_w + 8  # 右列 x 起始
    y_s = img_h + 8

    put_text(canvas, "State (24-dim):", (rx, y_s + line_h), font_scale=0.42, color=(0, 255, 255))

    # 分组显示
    state_groups = [
        ("Gripper L/R (0-5):", [
            f"  L: {state[0]:+.4f} {state[1]:+.4f} {state[2]:+.4f}",
            f"  R: {state[3]:+.4f} {state[4]:+.4f} {state[5]:+.4f}",
        ]),
        ("Joint pos (6-14):", [
            f"  j0={state[6]:+.3f} j1={state[7]:+.3f} j2={state[8]:+.3f}",
            f"  j3={state[9]:+.3f} j4={state[10]:+.3f} j5={state[11]:+.3f}",
            f"  j6={state[12]:+.3f} j7={state[13]:+.3f} j8={state[14]:+.3f}",
        ]),
        ("EEF pos (15-17):", [
            f"  x={state[15]:+.4f} y={state[16]:+.4f} z={state[17]:+.4f}",
        ]),
        ("EEF vel (18-20):", [
            f"  vx={state[18]:+.4f} vy={state[19]:+.4f} vz={state[20]:+.4f}",
        ]),
        ("EEF angvel (21-23):", [
            f"  wx={state[21]:+.4f} wy={state[22]:+.4f} wz={state[23]:+.4f}",
        ]),
    ]

    y_cur = y_s + line_h * 2 + 4
    for title, lines in state_groups:
        put_text(canvas, title, (rx, y_cur), font_scale=0.36, color=(180, 255, 180))
        for line in lines:
            y_cur += line_h
            put_text(canvas, line, (rx, y_cur), font_scale=0.33, color=(200, 200, 200))
        y_cur += line_h + 2

    return canvas


def visualize_trajectory(traj_id, fps=5):
    pt_path = DATA_DIR / f"trajectory_{traj_id}_demo_expert.pt"
    if not pt_path.exists():
        print(f"[WARN] {pt_path} not found, skipping.")
        return

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    actions = data["actions"][:, 0, :]                          # (T, 12)
    states = data["curr_obs"]["states"][:, 0, :]                # (T, 24)
    main_imgs = data["curr_obs"]["main_images"][:, 0]           # (T, 128, 128, 3)
    wrist_imgs = data["curr_obs"]["extra_view_images"][:, 0, 0] # (T, 128, 128, 3)
    rewards = data["rewards"][:, 0, 0]                          # (T,)

    total_steps = actions.shape[0]
    reward_sum = rewards.sum().item()
    status = "SUCCESS" if reward_sum > 0 else "FAIL"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"traj_{traj_id}_{status}.mp4"

    # 获取 canvas 尺寸
    sample_frame = render_frame(
        main_imgs[0].numpy(), wrist_imgs[0].numpy(),
        actions[0].numpy(), states[0].numpy(),
        0, total_steps, 0.0,
    )
    h, w = sample_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for t in range(total_steps):
        main_bgr = cv2.cvtColor(main_imgs[t].numpy(), cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_imgs[t].numpy(), cv2.COLOR_RGB2BGR)

        frame = render_frame(
            main_bgr, wrist_bgr,
            actions[t].numpy(), states[t].numpy(),
            t, total_steps, rewards[t].item(),
        )
        writer.write(frame)

    writer.release()
    print(f"[OK] traj_{traj_id} ({status}, {total_steps} steps) -> {out_path}")


def get_success_ids():
    ids = []
    for i in range(152):
        pt_path = DATA_DIR / f"trajectory_{i}_demo_expert.pt"
        if pt_path.exists():
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if data["rewards"].sum().item() > 0:
                ids.append(i)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Visualize robot trajectories")
    parser.add_argument("--traj_id", type=int, nargs="+", default=None, help="Trajectory IDs to visualize")
    parser.add_argument("--all", action="store_true", help="Visualize all successful trajectories")
    parser.add_argument("--fps", type=int, default=5, help="Video FPS (default: 5)")
    args = parser.parse_args()

    if args.all:
        traj_ids = get_success_ids()
        print(f"Found {len(traj_ids)} successful trajectories")
    elif args.traj_id is not None:
        traj_ids = args.traj_id
    else:
        traj_ids = [0]

    for tid in traj_ids:
        visualize_trajectory(tid, fps=args.fps)

    print(f"\nDone! Videos saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
