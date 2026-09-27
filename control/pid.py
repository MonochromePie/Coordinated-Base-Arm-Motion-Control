class PID:
    def __init__(self, kp: float, ki: float, kd: float, max_integral: float, max_output: float = 0.75):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        if self.max_output is not None:
            if abs(output) > abs(self.max_output):
                return  self.max_output * (abs(output) / output)
            else:
                return output

        return output

    def reset(self):
        self.integral = 0
        self.prev_error = 0