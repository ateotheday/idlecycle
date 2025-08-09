import cv2
import mediapipe as mp
import random
import time
import os
import webbrowser
import numpy as np
import streamlit as st

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Helper functions
def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    # Thumb
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
    return fingers

def detect_rps_gesture(fingers_list):
    s = sum(fingers_list)
    if s == 0:
        return "rock"
    elif s == 5:
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
    path = os.path.abspath("ggg.html")
    webbrowser.open(f"file://{path}")

# Initialize or reset session state variables
def init_session():
    if "cap" not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
    if "hands" not in st.session_state:
        st.session_state.hands = mp_hands.Hands(max_num_hands=1,
                                                min_detection_confidence=0.7,
                                                min_tracking_confidence=0.7)
    if "mode" not in st.session_state:
        st.session_state.mode = None
    if "selected_mode_stable" not in st.session_state:
        st.session_state.selected_mode_stable = 0
    if "prev_finger_count" not in st.session_state:
        st.session_state.prev_finger_count = -1
    if "pomodoro_cycles_completed" not in st.session_state:
        st.session_state.pomodoro_cycles_completed = 0

    # RPS game state
    for key in ["user_score", "comp_score", "last_result", "wait_for_next", "wait_start", "round_delay", "comp_move", "user_move"]:
        if key not in st.session_state:
            st.session_state[key] = 0 if 'score' in key else "" if 'result' in key else False if 'wait' in key else None

    # Racing game state
    for key in ["position", "obstacles", "score", "spawn_timer", "game_over", "frame_count"]:
        if key not in st.session_state:
            st.session_state[key] = 0 if key in ["position","score","spawn_timer","frame_count"] else False if key == "game_over" else []

# Mode selection screen
def choose_mode():
    cap = st.session_state.cap
    hands = st.session_state.hands

    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read from webcam")
        return None

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_landmarks)
            finger_count = sum(fingers)

    # Detect stable count for 2 or 3 fingers
    if finger_count == st.session_state.prev_finger_count and finger_count in [2,3]:
        st.session_state.selected_mode_stable += 1
    else:
        st.session_state.selected_mode_stable = 0

    st.session_state.prev_finger_count = finger_count

    # Display frame and instructions
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    st.write("Show 2 fingers for Rock-Paper-Scissors")
    st.write("Show 3 fingers for Racing Game")
    st.write(f"Detected fingers: {finger_count}")

    if st.session_state.selected_mode_stable > 30:
        st.session_state.mode = finger_count
        st.session_state.selected_mode_stable = 0
        return finger_count

    return None

# Rock Paper Scissors game step
def rps_game_step():
    cap = st.session_state.cap
    hands = st.session_state.hands

    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read from webcam")
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    user_move = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_landmarks)
            user_move = detect_rps_gesture(fingers)

    current_time = time.time()

    if not st.session_state.wait_for_next and user_move:
        comp_move = random.choice(["rock", "paper", "scissors"])
        winner = get_winner(user_move, comp_move)
        st.session_state.last_result = f"You: {user_move} | Computer: {comp_move} => {winner}"
        if winner == "You Win!":
            st.session_state.user_score += 1
        elif winner == "Computer Wins!":
            st.session_state.comp_score += 1
        st.session_state.wait_for_next = True
        st.session_state.wait_start = current_time
        st.session_state.comp_move = comp_move
        st.session_state.user_move = user_move

    if st.session_state.wait_for_next:
        elapsed = current_time - st.session_state.wait_start
        cv2.putText(frame, f"Next round in {int(st.session_state.round_delay - elapsed)}s", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if elapsed >= st.session_state.round_delay:
            st.session_state.wait_for_next = False
            st.session_state.last_result = ""

    cv2.putText(frame, f"Pomodoro Progress: {st.session_state.pomodoro_cycles_completed}/4 cycles", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
    cv2.putText(frame, "Rock-Paper-Scissors Gesture Game", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if st.session_state.user_move:
        cv2.putText(frame, f"Your Move: {st.session_state.user_move}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if st.session_state.comp_move:
        cv2.putText(frame, f"Computer Move: {st.session_state.comp_move}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if st.session_state.last_result:
        cv2.putText(frame, st.session_state.last_result, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Score You: {st.session_state.user_score} Computer: {st.session_state.comp_score}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Check winning conditions
    if st.session_state.user_score >= 10:
        st.session_state.pomodoro_cycles_completed += 1
        st.session_state.user_score = 0
        st.session_state.comp_score = 0
        st.session_state.wait_for_next = False
        st.success(f"Pomodoro cycle {st.session_state.pomodoro_cycles_completed}/4 complete! Take a 1 min break.")
        open_calendar()
        time.sleep(60)  # Caution: freezes Streamlit UI - consider alternatives

    if st.session_state.comp_score >= 10:
        st.warning("Computer Wins! Try again.")
        st.session_state.user_score = 0
        st.session_state.comp_score = 0
        st.session_state.wait_for_next = False

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# Racing Game helpers
def draw_road_perspective(frame, w, h, lanes, frame_num):
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
    cv2.ellipse(frame, (x, y + 60), (40, 15), 0, 0, 360, shadow_color, -1)
    pts = np.array([
        [x - 40, y - 50],
        [x + 40, y - 50],
        [x + 40, y + 50],
        [x - 40, y + 50]
    ])
    cv2.polylines(frame, [pts], True, (0, 150, 0), 3)
    cv2.fillPoly(frame, [pts], car_color)
    cv2.rectangle(frame, (x - 25, y - 45), (x + 25, y), (140, 255, 140), -1)
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

def draw_speedometer(frame, score, w, h):
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

def draw_hud(frame, cycle, cycle_progress, w, h, max_score, max_cycles):
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
    cv2.putText(frame, "Press 'Restart' or 'Quit'", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

# Racing game step
def racing_game_step():
    cap = st.session_state.cap
    hands = st.session_state.hands

    h, w = 480, 640
    lanes = 3
    lane_positions = [int(x) for x in np.linspace(150, 490, lanes)]
    speed = 10
    max_score = 10
    max_cycles = 4

    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read from webcam")
        return

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingers_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_landmarks)
            fingers_count = sum(fingers)

    if not st.session_state.game_over:
        if fingers_count == 1:
            st.session_state.position = 0
        elif fingers_count == 2:
            st.session_state.position = 1
        elif fingers_count == 3:
            st.session_state.position = 2

    draw_road_perspective(frame, w, h, lanes, st.session_state.frame_count)
    car_x = lane_positions[st.session_state.position]
    car_y = h - 110
    draw_car(frame, car_x, car_y)

    if not st.session_state.game_over:
        st.session_state.spawn_timer += 1
        if st.session_state.spawn_timer > 30:
            st.session_state.obstacles.append([random.choice(lane_positions), -60])
            st.session_state.spawn_timer = 0

        new_obstacles = []
        for obs in st.session_state.obstacles:
            obs[1] += speed
            if obs[1] < h + 70:
                new_obstacles.append(obs)
                draw_obstacle(frame, obs[0], obs[1])
                if abs(obs[0] - car_x) < 60 and abs(obs[1] - car_y) < 100:
                    st.session_state.game_over = True
            else:
                st.session_state.score += 1
        st.session_state.obstacles = new_obstacles

    cycle = st.session_state.score // max_score
    cycle_progress = st.session_state.score % max_score
    if cycle >= max_cycles:
        cv2.putText(frame, "Congrats! Pomodoro cycle complete!", (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        st.balloons()
    else:
        draw_hud(frame, cycle, cycle_progress, w, h, max_score, max_cycles)

    if st.session_state.game_over:
        cv2.putText(frame, "Game Over! Press Restart.", (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    st.image(frame, channels="BGR")

# Main Streamlit app
def main():
    st.title("🎮 Gesture-Based Pomodoro Games")
    init_session()

    if st.session_state.mode is None:
        st.write("Select Game Mode by showing 2 or 3 fingers steadily for a few seconds.")
        mode = choose_mode()
        if mode == 2:
            st.success("Mode Selected: Rock-Paper-Scissors Game")
            st.session_state.mode = 2
        elif mode == 3:
            st.success("Mode Selected: Racing Game")
            st.session_state.mode = 3
    elif st.session_state.mode == 2:
        st.write("### Rock-Paper-Scissors Game (Score 10 to complete cycle)")
        rps_game_step()
        if st.button("Restart"):
            st.session_state.user_score = 0
            st.session_state.comp_score = 0
            st.session_state.wait_for_next = False
            st.session_state.last_result = ""
            st.session_state.pomodoro_cycles_completed = 0
        if st.button("Quit to Mode Select"):
            st.session_state.mode = None
    elif st.session_state.mode == 3:
        st.write("### Racing Game (Score 40 to complete 4 cycles)")
        racing_game_step()
        if st.button("Restart"):
            st.session_state.position = 1
            st.session_state.obstacles = []
            st.session_state.score = 0
            st.session_state.spawn_timer = 0
            st.session_state.game_over = False
            st.session_state.frame_count = 0
            st.session_state.pomodoro_cycles_completed = 0
        if st.button("Quit to Mode Select"):
            st.session_state.mode = None

if __name__ == "__main__":
    main()
