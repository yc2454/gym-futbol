class Field:
    def __init__(self, origin_x, origin_y, width, height, goal_width):
        self.width = width
        self.height = height
        self.left_up_goal_tip = (origin_x, origin_y + (height/2) - (goal_width/2))
        self.left_down_goal_tip = (origin_x, origin_y + (height/2) + (goal_width/2))
        self.right_up_goal_tip = (origin_x+width, origin_y + (width/2) - (goal_width/2))
        self.right_down_goal_tip = (origin_x+width, origin_y + (width/2) + (goal_width/2))