import cv2
import mediapipe as mp
import random
import time
import webbrowser
import os
import numpy as np

pomodoro_cycles_completed = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (check x for right hand)
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1,5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def detect_rps_gesture(fingers_list):
    # Detect rock, paper, scissors based on fingers
    if sum(fingers_list) == 0:
        return "rock"
    elif sum(fingers_list) == 5:
        return "paper"
    elif fingers_list[1] == 1 and fingers_list[2] == 1 and fingers_list[0] == 0 and fingers_list[3] == 0 and fingers_list[4] == 0:
        return "scissors"
    else:
        return None

def get_winner(user_move, comp_move):
    if user_move == comp_move:
        return "Tie"
    elif (user_move == "rock" and comp_move == "scissors") or \
         (user_move == "paper" and comp_move == "rock") or \
         (user_move == "scissors" and comp_move == "paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

def open_calendar():
    # Update path to your actual calendar html file location here
    path = os.path.abspath("ggg.html")  # your file path
    webbrowser.open(f"file://{path}")

def rps_game(pomodoro_cycles_completed):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam error")
        return pomodoro_cycles_completed

    user_score = 0
    comp_score = 0
    last_result = ""
    comp_move = ""
    user_move = None
    wait_for_next = False
    wait_start = 0
    round_delay = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        user_move = None
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = []
                tips_ids = [4,8,12,16,20]
                if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for i in range(1,5):
                    if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i]-2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                user_move = detect_rps_gesture(fingers)

        current_time = time.time()

        if not wait_for_next and user_move:
            comp_move = random.choice(["rock", "paper", "scissors"])
            winner = get_winner(user_move, comp_move)
            last_result = f"You: {user_move} | Computer: {comp_move} => {winner}"
            if winner == "You Win!":
                user_score += 1
            elif winner == "Computer Wins!":
                comp_score += 1
            wait_for_next = True
            wait_start = current_time

        if wait_for_next:
            elapsed = current_time - wait_start
            cv2.putText(frame, f"Next round in {int(round_delay - elapsed)}s", (10, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if elapsed >= round_delay:
                wait_for_next = False
                last_result = ""

        # Show current pomodoro cycle progress on screen
        cv2.putText(frame, f"Pomodoro Progress: {pomodoro_cycles_completed}/4 cycles", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)

        cv2.putText(frame, "Rock-Paper-Scissors Gesture Game", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if user_move:
            cv2.putText(frame, f"Your Move: {user_move}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if comp_move:
            cv2.putText(frame, f"Computer Move: {comp_move}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if last_result:
            cv2.putText(frame, last_result, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Score You: {user_score} Computer: {comp_score}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Check winning condition (score out of 10)
        if user_score >= 10:
            pomodoro_cycles_completed += 1

            if pomodoro_cycles_completed == 4:
                # Completed 4 cycles - Pomodoro complete message
                cv2.putText(frame, "🎉 Pomodoro Cycles Complete! 🎉", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                cv2.imshow("RPS Game", frame)
                cv2.waitKey(5000)
                pomodoro_cycles_completed = 0  # reset after completion
            else:
                cv2.putText(frame, f"Cycle {pomodoro_cycles_completed}/4 complete! Take a 1 min break.", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("RPS Game", frame)
                cv2.waitKey(3000)

            cap.release()
            cv2.destroyAllWindows()

            # Open calendar page for 1 min break
            open_calendar()
            time.sleep(60)

            return pomodoro_cycles_completed

        if comp_score >= 10:
            cv2.putText(frame, "Computer Wins! Replay Game.", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("RPS Game", frame)
            cv2.waitKey(3000)
            user_score = 0
            comp_score = 0
            last_result = ""
            comp_move = ""
            wait_for_next = False
            continue

        cv2.imshow("RPS Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return pomodoro_cycles_completed


def racing_game():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam error")
        return

    position = 1  # Start in middle lane (0-left, 1-middle, 2-right)
    lanes = 3
    lane_positions = [int(x) for x in np.linspace(150, 490, lanes)]  # evenly spaced lanes
    h, w = 480, 640
    speed = 10
    obstacles = []
    score = 0
    spawn_timer = 0
    game_over = False
    frame_count = 0

    max_score = 10  # changed from 15 to 10
    max_cycles = 4

    def draw_road_perspective(frame, frame_num):
        horizon_y = 100
        vanishing_x = w // 2
        road_width_bottom = w
        road_width_top = w // 4

        overlay = frame.copy()
        pts = np.array([
            [vanishing_x - road_width_top//2, horizon_y],
            [vanishing_x + road_width_top//2, horizon_y],
            [vanishing_x + road_width_bottom//2, h],
            [vanishing_x - road_width_bottom//2, h]
        ])
        cv2.fillPoly(overlay, [pts], (30, 30, 30))
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        dash_length = 30
        gap_length = 30
        for lane_x_ratio in np.linspace(-0.33, 0.33, lanes + 1):
            points = []
            for y in range(horizon_y, h, dash_length + gap_length):
                interp = (y - horizon_y) / (h - horizon_y)
                top_x = vanishing_x + lane_x_ratio * road_width_top / 2
                bottom_x = vanishing_x + lane_x_ratio * road_width_bottom / 2
                x = int(top_x + interp * (bottom_x - top_x))
                points.append((x, y))
            for i in range(0, len(points), 2):
                if i + 1 < len(points):
                    cv2.line(frame, points[i], points[i+1], (255, 255, 255), 3, lineType=cv2.LINE_AA)

    def draw_car(frame, x, y):
        car_color = (0, 180, 0)
        shadow_color = (20, 20, 20)
        thickness = -1
        cv2.ellipse(frame, (x, y + 60), (40, 15), 0, 0, 360, shadow_color, -1)
        pts = np.array([
            [x - 40, y - 50],
            [x + 40, y - 50],
            [x + 40, y + 50],
            [x - 40, y + 50]
        ])
        cv2.polylines(frame, [pts], True, (0, 150, 0), 3)
        cv2.fillPoly(frame, [pts], car_color)
        cv2.rectangle(frame, (x - 25, y - 45), (x + 25, y), (140, 255, 140), thickness)
        cv2.circle(frame, (x - 25, y + 55), 20, (40, 40, 40), -1)
        cv2.circle(frame, (x + 25, y + 55), 20, (40, 40, 40), -1)
        cv2.circle(frame, (x - 25, y + 55), 15, (90, 90, 90), 3)
        cv2.circle(frame, (x + 25, y + 55), 15, (90, 90, 90), 3)

    def draw_obstacle(frame, x, y):
        base_width = 40
        height = 60
        pts = np.array([
            [x, y],
            [x - base_width // 2, y + height],
            [x + base_width // 2, y + height]
        ])
        cv2.fillPoly(frame, [pts], (0, 140, 255))
        stripe_height = height // 5
        for i in range(0, height, stripe_height*2):
            stripe_pts = np.array([
                [x - base_width // 2 + 5, y + i],
                [x + base_width // 2 - 5, y + i],
                [x + base_width // 2 - 5, y + i + stripe_height],
                [x - base_width // 2 + 5, y + i + stripe_height]
            ])
            cv2.fillPoly(frame, [stripe_pts], (255, 255, 255))

    def draw_speedometer(frame, score):
        center = (w - 100, h - 100)
        radius = 70
        thickness = 20
        cv2.circle(frame, center, radius, (100, 100, 100), thickness)
        angle = int(min(score, 100) * 1.8)
        for i in range(angle):
            color_val = 255 - int(i * 255 / 180)
            cv2.ellipse(frame, center, (radius, radius), 180, 0, i, (0, color_val, 0), thickness)
        cv2.putText(frame, f"Score: {score}", (w - 160, h - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    def draw_hud(frame, cycle, cycle_progress):
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (w - 5, 110), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "Use 1/2/3 fingers to move Left/Mid/Right", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Cycle: {cycle}/4", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {cycle_progress}/{max_score}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'r' to Restart, 'q' to Quit", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (w, h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        fingers_count = 0
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks)

        if not game_over:
            if fingers_count == 1:
                position = 0
            elif fingers_count == 2:
                position = 1
            elif fingers_count == 3:
                position = 2

        draw_road_perspective(frame, frame_count)
        car_x = lane_positions[position]
        car_y = h - 110
        draw_car(frame, car_x, car_y)

        if not game_over:
            spawn_timer += 1
            if spawn_timer > 30:
                obstacles.append([random.choice(lane_positions), -60])
                spawn_timer = 0

            new_obstacles = []
            for obs in obstacles:
                obs[1] += speed
                if obs[1] < h + 70:
                    new_obstacles.append(obs)
                    draw_obstacle(frame, obs[0], obs[1])
                    if abs(obs[0] - car_x) < 60 and abs(obs[1] - car_y) < 100:
                        game_over = True
                else:
                    score += 1
            obstacles = new_obstacles

        cycle = score // max_score
        cycle_progress = score % max_score
        if cycle >= max_cycles:
            cv2.putText(frame, "Congrats! Pomodoro cycle complete!", (50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("Racing Game", frame)
            cv2.waitKey(3000)
            break

        draw_hud(frame, min(cycle, max_cycles), cycle_progress)
        draw_speedometer(frame, cycle_progress)

        if game_over:
            cv2.putText(frame, "GAME OVER! Press 'r' to Restart or 'q' to Quit", (50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imshow("Racing Game", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if game_over and key == ord('r'):
            obstacles.clear()
            score = 0
            spawn_timer = 0
            game_over = False
            position = 1
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()

def choose_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam error")
        return

    print("Show 2 or 3 fingers to select mode:")
    print("2 - Rock-Paper-Scissors Game")
    print("3 - Simple Racing Game")

    selected_mode = None
    stable_count = 0
    prev_finger_count = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        finger_count = 0
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(hand_landmarks)

        cv2.putText(frame, "Show 2 or 3 fingers to choose mode", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detected fingers: {finger_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Choose Mode", frame)

        if finger_count == prev_finger_count and finger_count in [2, 3]:
            stable_count += 1
        else:
            stable_count = 0
        prev_finger_count = finger_count

        if stable_count > 60:
            selected_mode = finger_count
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return selected_mode

def main():
    pomodoro_cycles_completed = 0
    while True:
        mode = choose_mode()
        if mode == 2:
            print("Starting Rock-Paper-Scissors game...")
            pomodoro_cycles_completed = rps_game(pomodoro_cycles_completed)
            print(f"Pomodoro cycles completed: {pomodoro_cycles_completed}/4")
        elif mode == 3:
            print("Starting Racing Game...")
            racing_game()
        else:
            print("No valid mode selected, exiting.")
            break


if __name__ == "__main__":
    main()
