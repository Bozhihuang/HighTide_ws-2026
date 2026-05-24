# Current mission and task state
uint8 IDLE=0
uint8 PRE_DIVE=1
uint8 TASK_GATE=2
uint8 TASK_SLALOM=3
uint8 TASK_BINS=4
uint8 TASK_TORPEDOES=5
uint8 TASK_OCTAGON=6
uint8 RETURN_HOME=7
uint8 EMERGENCY=8
uint8 COMPLETED=9

uint8 current_task
string current_behavior
string status_message
builtin_interfaces/Time task_start_time
string chosen_role              # 'survey_repair' or 'search_rescue'
