"""my_controller controller."""
# A multimodal Artificial Intelligence Architecture for Autonomus Robot Navigation
#  from controller import Robot, Motor, DistanceSensor


# PROJECT OVERVIEW

#   Problem statement - A single robot that will navigate and solve the maze using signs placed along to reach the exit 
#   Objectives - The epuck searches then spots a sign
                # uses OCR and NLP to decode the sign
                #converts it to command and approaches the sign
                #speaks the command and performs the action
                #repeats process till maze is complete
                
#   Overview of modules:
#   CV, OCR(Deep learning), NLP, Speech, Robotics Control


# IMPORTS AND DEPENDENCIES

# Webots 
# OpenCV
# Easyocr (OCR)
# NLP 
# Speech libraries
# NumPy

print("[INIT] Importing libraries...")

from controller import Robot, Camera, Speaker, GPS, Motor, DistanceSensor

import math
import numpy as np
import cv2
import easyocr

print("[INIT] Libraries loaded successfully")


 
# DEBUG SYSTEM
 

DEBUG = True

def log(tag, msg):
    if DEBUG:
        print(f"{tag} {msg}")


 
# ROBOT INITIALIZATION
 

print("[INIT] Initializing robot...")

robot = Robot()
timestep = int(robot.getBasicTimeStep())
MAX_SPEED = 5.28

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

speaker = robot.getDevice('speaker')
speaker.setLanguage('en-UK')

camera = robot.getDevice('camera')
camera.enable(timestep)

imu = robot.getDevice('inertial unit')
imu.enable(timestep)

emitter = robot.getDevice("emitter(1)")

gps = robot.getDevice("gps")
gps.enable(timestep)

 
# PROXIMITY SENSORS
 

ps = []
psNames = ['ps0','ps1','ps2','ps3','ps4','ps5','ps6','ps7']

for i in range(8):
    sensor = robot.getDevice(psNames[i])
    sensor.enable(timestep)
    ps.append(sensor)


 
# OCR
 

print("[INIT] Loading Deep learning model...")

reader = easyocr.Reader(['en'], gpu=False)

print("[INIT] Deep learning model loaded")


 
# STATE
 

state = "SEARCH"

stable_command = None
command_locked = False
pending_turn = None


TURN_TOLERANCE = 0.05


last_executed_command = None
active_command = None
turn_counter = 0
MAX_TURNS_BEFORE_RESET = 5

 
# CAMERA
 

def capture_image():
    img = camera.getImage()
    if img is None:
        return None

    w = camera.getWidth()
    h = camera.getHeight()

    frame = np.frombuffer(img, np.uint8).reshape((h, w, 4))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


 
# SENSOR HELPERS
 

def get_yaw():
    return imu.getRollPitchYaw()[2]


def angle_diff(target, current):
    diff = target - current
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def wall_detected():
    v = [ps[i].getValue() for i in range(8)]
    return v[0] > 80 or v[1] > 80 or v[6] > 80 or v[7] > 80


 
# OCR(DEEP LEARNING) + POSITION
 

def extract_text_and_position(image):
    results = reader.readtext(image)

    if not results:
        return "", None

    filtered = [r for r in results if r[2] > 0.5]

    if not filtered:
        return "", None

    text = " ".join([r[1] for r in filtered]).strip()

    x_vals = []
    for box, _, _ in filtered:
        for p in box:
            x_vals.append(p[0])

    x_center = sum(x_vals) / len(x_vals) if x_vals else None

    return text, x_center


 
# NLP
 

def interpret_text(text):
    if not text:
        return None

    t = text.upper().replace(" ", "")

    if "RIGHT"[:3] in t or "RIGH" in t or "GHT" in t or "HT" in t:
        return "TURNING RIGHT"

    if "LEFT"[:3] in t or "LE" in t or "EFT" in t:
        return "TURNING LEFT"

    if "DO NOT ENTER"[:3] in t or "ENTER" in t:
        return "NO ENTRY"

    if "EXIT" in t:
        return "FINAL STOP MAZE COMPLETE"

    return None


 
# SPEECH
 

def speak_command(text):
    if text:
        log("[SPEAKER]", f"Speaking AND {text}")
        speaker.speak(text, 1.0)


 
# STUCK + RECOVERY SYSTEM
 
stuck_counter = 0
MAX_STUCK_STEPS = 40

STUCK_THRESHOLD = 60   # tune depending on timestep



target_yaw = None
forward_after_turn = 0


def rad_to_deg(r):
    return r * 180.0 / math.pi

# MOVEMENT SYSTEM 


def move_robot():
    global state, pending_turn, stable_command, command_locked
    global last_executed_command
    global active_command, target_yaw, forward_after_turn
    global turn_counter


    # FINAL STOP STATE (MAZE COMPLETE)

    if state == "STOP":
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        log("[STATE]", "STOPPED AT EXIT")
        return

    log("[STATE]", state)

    ps_values = [ps[i].getValue() for i in range(8)]
    front_wall = ps_values[0] > 100 or ps_values[7] > 100


      
    # EXECUTE TURN
      
    if state == "EXECUTE_TURN":

        current = get_yaw()

        if target_yaw is None:

            if pending_turn == "TURNING LEFT":
                target_yaw = current + math.pi / 2
            elif pending_turn == "TURNING RIGHT":
                target_yaw = current - math.pi / 2
            elif pending_turn == "NO ENTRY":
                target_yaw = current + math.pi
            else:
                target_yaw = current

            log("[TURN INIT]", f"target = {rad_to_deg(target_yaw):.2f}°")

        error = angle_diff(target_yaw, current)

        log("[TURN]", f"error = {rad_to_deg(error):.2f}°")

        # DONE TURNING
        if abs(error) < TURN_TOLERANCE:

            leftMotor.setVelocity(0)
            rightMotor.setVelocity(0)

            log("[STATE]", "TURN COMPLETE AND FORWARD")

            last_executed_command = pending_turn
            
            # Turn Counter
            
            global turn_counter
            turn_counter += 1
            log("[SYSTEM]", f"Turn count = {turn_counter}")

            forward_after_turn = 15
            state = "FORWARD_AFTER_TURN"

            return

        # TURN MOTION
        if error > 0:
            leftMotor.setVelocity(-0.3 * MAX_SPEED)
            rightMotor.setVelocity(0.3 * MAX_SPEED)
        else:
            leftMotor.setVelocity(0.3 * MAX_SPEED)
            rightMotor.setVelocity(-0.3 * MAX_SPEED)

        return

      
    # FORWARD AFTER TURN 
      
    if state == "FORWARD_AFTER_TURN":
    
        front = ps_values[0] > 80 or ps_values[7] > 80
        left_side = ps_values[5] > 80
        right_side = ps_values[2] > 80
        
        log("[FORWARD]", f"F={front} L={left_side} R={right_side}")
    
          
        # Wall ahead steering
          
        if front:
            log("[FORWARD]", "Wall ahead AND adjusting")
    
            if pending_turn == "TURNING LEFT":
                leftMotor.setVelocity(0.2 * MAX_SPEED)
                rightMotor.setVelocity(0.6 * MAX_SPEED)
            elif pending_turn == "TURNING RIGHT":
                leftMotor.setVelocity(0.6 * MAX_SPEED)
                rightMotor.setVelocity(0.2 * MAX_SPEED)
            else:
                leftMotor.setVelocity(-0.2 * MAX_SPEED)
                rightMotor.setVelocity(0.2 * MAX_SPEED)
    
            return
    
          
        #  Controlled forward exit
          
        if forward_after_turn > 0:
            forward_after_turn -= 1
    
            leftMotor.setVelocity(0.5 * MAX_SPEED)
            rightMotor.setVelocity(0.5 * MAX_SPEED)
            return
    
          
        # Reset only when clear
          
        if not front:
            log("[STATE]", "EXIT COMPLETE AND SEARCH")
    
            pending_turn = None
            active_command = None
            command_locked = False
            stable_command = None
            target_yaw = None
    
            state = "SEARCH"
        
        return
    
    
      
    # APPROACH SIGN
      
    if state == "APPROACH_SIGN":

        near_sign = ps_values[0] > 120 or ps_values[7] > 120

        log("[APPROACH]", f"PS0={ps_values[0]:.1f} PS7={ps_values[7]:.1f}")

        # CLOSE AND ACT
        if near_sign:

            log("[STATE]", f"REACHED SIGN AND {pending_turn}")

            speak_command(pending_turn)

            # MAZE COMPLETE CASE
            if pending_turn == "FINAL STOP MAZE COMPLETE":
                log("[STATE]", "EXIT REACHED AND STOPPING")
                state = "STOP"
                return

            state = "EXECUTE_TURN"
            return

        # MOVE FORWARD
        leftMotor.setVelocity(0.8 * MAX_SPEED)
        rightMotor.setVelocity(0.8 * MAX_SPEED)
        return


    
 
# MAIN LOOP 
 

step_counter = 0
stuck_counter = 0

print("[SYSTEM] Starting loop...")

while robot.step(timestep) != -1:

    step_counter += 1

    ps_values = [ps[i].getValue() for i in range(8)]
    near_previous_sign = ps_values[0] > 80 or ps_values[7] > 80
    front_wall = ps_values[0] > 100 or ps_values[7] > 100

      
    # SOFT RESET 
      
    if turn_counter >= MAX_TURNS_BEFORE_RESET and state != "STOP":

        log("[SYSTEM]", "SOFT RESET TRIGGERED")

        pending_turn = None
        stable_command = None
        command_locked = False
        target_yaw = None
        active_command = None
        last_executed_command = None

        turn_counter = 0
        stuck_counter = 0
        state = "SEARCH"

        continue

      
    # STUCK DETECTION
      
    if state == "SEARCH":

        if front_wall:
            stuck_counter += 1
        else:
            stuck_counter = max(0, stuck_counter - 1)

        if stuck_counter > MAX_STUCK_STEPS:

            log("[RECOVERY]", "Robot stuck AND escape")

            # reverse
            leftMotor.setVelocity(-0.3 * MAX_SPEED)
            rightMotor.setVelocity(-0.3 * MAX_SPEED)
            for _ in range(10):
                robot.step(timestep)

            # rotate
            leftMotor.setVelocity(0.3 * MAX_SPEED)
            rightMotor.setVelocity(-0.3 * MAX_SPEED)
            for _ in range(15):
                robot.step(timestep)

            stuck_counter = 0
            continue

      
    # VISION
      
    if state == "SEARCH" and step_counter % 10 == 0:

        img = capture_image()
        if img is None:
            move_robot()
            continue

        text, _ = extract_text_and_position(img)
        command = interpret_text(text)

        log("[VISION]", f"text={text} AND command={command}")

        # unlock memory when far
        if last_executed_command and not near_previous_sign:
            command_locked = False
            
        # HANDLE NO COMMAND
        if command is None:
            move_robot()
            continue
        

        # ignore same sign only if still close
        if command and command == last_executed_command and near_previous_sign:
            log("[MEMORY]", "Ignoring same sign")
            move_robot()
            continue

          
        #  LOCK SIGN OR EXIT
          
        if command:

            stable_command = command
            pending_turn = command
            command_locked = True

            log("[STATE]", f"LOCKED AND {command}")
            state = "APPROACH_SIGN"
            
            
      
    # SEND POSITION + STATE
     
    pos = gps.getValues()
    yaw = get_yaw()

    message = f"{pos[0]},{pos[2]},{yaw},{state}"
    emitter.send(message.encode("utf-8"))

                
   
    # ALWAYS MOVE
      
    move_robot()

 
# Side Note
  


# the system was redesigned as a single-agent autonomous navigation system. 
# This allowed focus on reliable perception-driven decision-making using CV, OCR(Deep learning) and NLP.

