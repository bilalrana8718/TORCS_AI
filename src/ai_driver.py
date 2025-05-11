import msgParser
import carState
import carControl
import joblib
import pandas as pd
import numpy as np
import os
import time

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PARENT_DIR, "models")

class AIDriver(object):
    def __init__(self, stage):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 200
        self.prev_rpm = None
        
        self.prediction_count = 0
        self.total_prediction_time = 0
        
        self.is_recovering = False
        self.recovery_start_time = 0
        self.recovery_duration = 2.0  
        self.race_started = False
        self.min_distance_for_recovery = 50.0 
        
        self.slow_accel_start_time = 0
        self.slow_accel_duration = 2.0  
        self.is_slow_accelerating = False
        
        self.recovery_cooldown_time = 0
        self.recovery_cooldown_duration = 2.0  
        
        self.is_acceleration_burst = False
        self.acceleration_burst_start_time = 0
        self.acceleration_burst_duration = 1.75  
        
        self.load_models()

    def load_models(self):
        '''Load the trained ML models and scalers'''
        try:
            print("Loading ML models and scalers...")
            
            model_path = os.path.join(MODELS_DIR, "racing_model.joblib")
            scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
            steering_scaler_path = os.path.join(MODELS_DIR, "steering_scaler.joblib")
            
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                print("Checking parent directory as fallback...")
                model_path = os.path.join(PARENT_DIR, "racing_model.joblib")
                
            if not os.path.exists(scaler_path):
                print(f"Warning: Scaler file not found at {scaler_path}")
                print("Checking parent directory as fallback...")
                scaler_path = os.path.join(PARENT_DIR, "scaler.joblib")
                
            if not os.path.exists(steering_scaler_path):
                print(f"Warning: Steering scaler file not found at {steering_scaler_path}")
                print("Checking parent directory as fallback...")
                steering_scaler_path = os.path.join(PARENT_DIR, "steering_scaler.joblib")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.steering_scaler = joblib.load(steering_scaler_path)
            
            print(f"Models and scalers loaded successfully from: {os.path.dirname(model_path)}")
            
            self.input_columns = [
                "Track_1", "Track_2", "Track_3", "Track_4", "Track_5", 
                "Track_6", "Track_7", "Track_8", "Track_9", "Track_10", 
                "Track_11", "Track_12", "Track_13", "Track_14", "Track_15",
                "Track_16", "Track_17", "Track_18", "Track_19",
                "SpeedX", "SpeedY", "SpeedZ", "Angle", "TrackPosition",
                "RPM", "WheelSpinVelocity_1", "WheelSpinVelocity_2",
                "WheelSpinVelocity_3", "WheelSpinVelocity_4", "DistanceCovered",
                "DistanceFromStart", "CurrentLapTime", "Damage",
                "Opponent_9", "Opponent_10", "Opponent_11", "Opponent_19"
            ]
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found! {e}")
            print("Please ensure the model files are in the 'r' directory or parent directory.")
            raise
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def init(self):
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        self.state.setFromMsg(msg)

        self.predict_and_control()
        return self.control.toMsg()
    
    def prepare_input_data(self):
        track = self.state.getTrack()
        if track is None or len(track) < 19:
            track = [200.0] * 19
        
        wheel_spin = self.state.getWheelSpinVel()
        if wheel_spin is None or len(wheel_spin) < 4:
            wheel_spin = [0.0] * 4
        
        opponents = self.state.getOpponents()
        if opponents is None or len(opponents) < 36:
            opponents = [200.0] * 36
        
        input_data = {
            "Track_1": track[0],
            "Track_2": track[1],
            "Track_3": track[2],
            "Track_4": track[3],
            "Track_5": track[4],
            "Track_6": track[5],
            "Track_7": track[6],
            "Track_8": track[7],
            "Track_9": track[8],
            "Track_10": track[9],
            "Track_11": track[10],
            "Track_12": track[11],
            "Track_13": track[12],
            "Track_14": track[13],
            "Track_15": track[14],
            "Track_16": track[15],
            "Track_17": track[16],
            "Track_18": track[17],
            "Track_19": track[18],

            "SpeedX": self.state.getSpeedX() or 0.0,
            "SpeedY": self.state.getSpeedY() or 0.0,
            "SpeedZ": self.state.getSpeedZ() or 0.0,
            "Angle": self.state.getAngle() or 0.0,
            "TrackPosition": self.state.getTrackPos() or 0.0,
            "RPM": self.state.getRpm() or 0.0,
            "WheelSpinVelocity_1": wheel_spin[0],
            "WheelSpinVelocity_2": wheel_spin[1],
            "WheelSpinVelocity_3": wheel_spin[2],
            "WheelSpinVelocity_4": wheel_spin[3],

            "DistanceCovered": self.state.getDistRaced() or 0.0,
            "DistanceFromStart": self.state.getDistFromStart() or 0.0,
            "CurrentLapTime": self.state.getCurLapTime() or 0.0,

            "Damage": self.state.getDamage() or 0.0,
            "Opponent_9": opponents[8],
            "Opponent_10": opponents[9],
            "Opponent_11": opponents[10],
            "Opponent_19": opponents[18],
        }
        
        return pd.DataFrame([input_data])
    
    def predict_and_control(self):
        try:
            start_time = time.time()
            
            self.check_if_stuck()
            
            self.check_zero_speed()
            
            if self.is_recovering:
                self.recovery_control()
                return
            
            if self.is_acceleration_burst:
                self.apply_acceleration_burst()
                return
            
            X_new = self.prepare_input_data()
            
            missing_columns = [col for col in self.input_columns if col not in X_new.columns]
            if missing_columns:
                print(f"Warning: Missing columns: {missing_columns}")
                for col in missing_columns:
                    X_new[col] = 0.0
            
            X_new = X_new[self.input_columns]
            
            X_new_scaled = self.scaler.transform(X_new)
            
            output = self.model.predict(X_new_scaled)
            
            output_df = pd.DataFrame(output, columns=["Steering", "Acceleration", "Braking"])
            
            output_df[["Steering"]] = self.steering_scaler.inverse_transform(output_df[["Steering"]])
            output_df["Steering"] = np.clip(output_df["Steering"].values[0], -1, 1)
            
            acceleration = 1.0 if output_df["Acceleration"].values[0] > 0.5 else 0.0
            braking = 1.0 if output_df["Braking"].values[0] > 0.5 else 0.0
            
            self.control.setSteer(output_df["Steering"].values[0])
            self.control.setAccel(acceleration)
            self.control.setBrake(braking)
            
            self.handle_gear()
            
            prediction_time = time.time() - start_time
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            avg_time = self.total_prediction_time / self.prediction_count
            
            if self.prediction_count % 100 == 0:
                print("--------------------------------")
                print(f"Made {self.prediction_count} predictions. Avg time: {avg_time:.4f}s")
                print(f"Steering: {output_df['Steering'].values[0]:.4f}, Acceleration: {acceleration}, Braking: {braking}")
                print(f"Speed: {self.state.getSpeedX():.1f}, RPM: {self.state.getRpm():.0f}, Gear: {self.state.getGear()}")
                print("--------------------------------")

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            self.simple_control()
    
    def handle_gear(self):
        '''Handle gear changes based on RPM'''
        rpm = self.state.getRpm() or 0.0
        gear = self.state.getGear() or 1
        speed = self.state.getSpeedX() or 0.0
        
        upshift_rpm = {
            1: 5500, 2: 6000, 3: 6500, 4: 7000, 5: 7500, 6: 8000
        }
        downshift_rpm = {
            2: 2500, 3: 3000, 4: 3000, 5: 3500, 6: 3500
        }
        
        if gear < 6 and rpm >= upshift_rpm.get(gear, 7000):
            self.control.setClutch(0.5)
            gear += 1
        elif gear > 1 and rpm <= downshift_rpm.get(gear, 3000):
            if not (gear > 2 and speed > 60 + (gear * 20)):
                self.control.setClutch(0.5)
                gear -= 1
        else:
            current_clutch = self.control.getClutch()
            if current_clutch > 0:
                self.control.setClutch(max(0, current_clutch - 0.1))
        
        self.control.setGear(gear)
    
    def simple_control(self):
        '''Simple fallback control logic if ML prediction fails'''
        angle = self.state.getAngle() or 0.0
        trackPos = self.state.getTrackPos() or 0.0
        self.control.setSteer((angle - trackPos*0.5)/self.steer_lock)
        
        speed = self.state.getSpeedX() or 0.0
        if speed < self.max_speed:
            self.control.setAccel(1.0)
            self.control.setBrake(0.0)
        else:
            self.control.setAccel(0.0)
            self.control.setBrake(0.1)
        
        self.handle_gear()
            
    def onShutDown(self):
        print(f"AI Driver shutdown. Made {self.prediction_count} predictions.")
        pass
    
    def onRestart(self):
        self.prediction_count = 0
        self.total_prediction_time = 0
        print("AI Driver restarted")
    
    def check_if_stuck(self):
        '''Simple check if the car is stuck or crashed'''
        current_speed = abs(self.state.getSpeedX() or 0.0)
        current_distance = self.state.getDistRaced() or 0.0
        current_time = time.time()
        
        if self.is_recovering or self.is_acceleration_burst:
            return
            
        if current_time < self.recovery_cooldown_time:
            return
            
        if current_distance > self.min_distance_for_recovery:
            self.race_started = True
        
        is_accelerating = self.control.getAccel() > 0.5
        is_slow = current_speed < 5.0
        
        if is_accelerating and is_slow and self.race_started:
            if not self.is_slow_accelerating:
                self.is_slow_accelerating = True
                self.slow_accel_start_time = current_time
                print(f"Car is accelerating but slow (speed: {current_speed:.1f}). Monitoring...")
            
            slow_accel_time = current_time - self.slow_accel_start_time
            
            if slow_accel_time >= self.slow_accel_duration:
                print(f"Car has been accelerating but slow for {slow_accel_time:.1f} seconds")
                print("Initiating recovery procedure...")
                self.is_recovering = True
                self.recovery_start_time = current_time
                self.is_slow_accelerating = False  # Reset the state
        else:
            if self.is_slow_accelerating:
                print("Car no longer in slow-accelerating state")
                self.is_slow_accelerating = False
    
    def check_zero_speed(self):
        '''Check if the car is completely stopped and trigger acceleration burst'''
        if self.is_recovering or self.is_acceleration_burst:
            return
            
        current_speed = abs(self.state.getSpeedX() or 0.0)
        
        current_time = time.time()
        if current_time < self.recovery_cooldown_time:
            return
        
        if current_speed < 0.5 and self.race_started: 
            print(f"Car is nearly stopped! Speed: {current_speed:.2f}. Applying acceleration burst...")
            self.is_acceleration_burst = True
            self.acceleration_burst_start_time = current_time
            return
    
    def recovery_control(self):
        '''Simple recovery control - just reverse for 2 seconds'''
        track_pos = self.state.getTrackPos() or 0.0
        
        steer_value = track_pos
        steer_value = np.clip(steer_value, -1.0, 1.0)
        
        self.control.setSteer(steer_value)
        self.control.setGear(-1) 
        self.control.setAccel(1.0)  
        self.control.setBrake(0.0)
        
        print(f"Recovery: Reversing with steer {steer_value:.2f}")
        
        current_time = time.time()
        if current_time - self.recovery_start_time > self.recovery_duration:
            print("Recovery completed, resuming normal driving")
            self.is_recovering = False
            
            self.recovery_cooldown_time = current_time + self.recovery_cooldown_duration
            print(f"Recovery cooldown active for {self.recovery_cooldown_duration} seconds")
    
    def apply_acceleration_burst(self):
        '''Apply a short burst of acceleration to get the car moving'''
        track_pos = self.state.getTrackPos() or 0.0
        current_time = time.time()
        burst_time = current_time - self.acceleration_burst_start_time
        
        steer_value = -track_pos * 0.5
        steer_value = np.clip(steer_value, -1.0, 1.0)
        
        self.control.setSteer(steer_value)
        self.control.setGear(1)  
        self.control.setAccel(1.0)  
        self.control.setBrake(0.0)
        self.control.setClutch(0.0)  
        
        if int(burst_time * 2) > int((burst_time - 0.01) * 2):  
            print(f"Acceleration burst: t={burst_time:.2f}s, Speed={self.state.getSpeedX():.2f}, Gear={self.state.getGear()}")
        
        if burst_time > self.acceleration_burst_duration:
            print(f"Acceleration burst completed after {burst_time:.2f}s, speed: {self.state.getSpeedX():.2f}")
            self.is_acceleration_burst = False
            
            self.recovery_cooldown_time = current_time + 1.0  