from pynput import keyboard

class KeyBoardCommand:
    def __init__(self):
        self.current_pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
    
    # 按下键的处理函数
    def on_press(self, key):
        try:
            self.current_pressed_keys.add(key)  # 将按键加入集合
        except Exception as e:
            print(f"Error: {e}")

    # 释放键的处理函数
    def on_release(self, key):
        try:
            if key in self.current_pressed_keys:
                self.current_pressed_keys.remove(key)  # 从集合中移除按键
        except Exception as e:
            print(f"Error: {e}")

    # 定义读取函数
    def get_current_pressed_keys(self):
        """
        返回当前按下的所有键。
        """
        return [str(key) for key in self.current_pressed_keys]

    def get_keyboard_to_robot(self):
        keys = self.get_current_pressed_keys()
        commands = []
        if "'q'" in keys:
            commands.append('down')
        if "'a'" in keys:
            commands.append('left')
        if "'w'" in keys:
            commands.append('forward')
        if "'s'" in keys:
            commands.append('backward')
        if "'e'" in keys:
            commands.append('up')
        if "'d'" in keys:
            commands.append('right')
            
        if "'u'" in keys:
            commands.append('roll_up')
        if "'j'" in keys:
            commands.append('roll_down')
        if "'i'" in keys:
            commands.append('pitch_up')
        if "'k'" in keys:
            commands.append('pitch_down')
        if "'o'" in keys:
            commands.append('yaw_up')
        if "'l'" in keys:
            commands.append('yaw_down')

        if "'y'" in keys:
            commands.append('gripper_open')
        if "'h'" in keys:
            commands.append('gripper_close')
            
        if "'z'" in keys:
            commands.append('stop')
        if "'x'" in keys:
            commands.append('drop')
        if "'n'" in keys:
            commands.append('record')
        if "'m'" in keys:
            commands.append('stop_record')
            
        if "'1'" in keys:
            commands.append('back_left')
        if "'2'" in keys:
            commands.append('back_right')
        if "'3'" in keys:
            commands.append('front_left')
        if "'4'" in keys:
            commands.append('front_right')
            
        return commands