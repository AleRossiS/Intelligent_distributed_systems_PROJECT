import config as cfg

def broadcast_data(broadcaster, robot_list):
    """
    Simulates a network broadcast. The transmitting robot packages its current 
    knowledge and sends it as a payload to all other robots in the system.
    """
    # 1. The broadcaster prepares its data packet (dictionary)
    payload = broadcaster.generate_payload()
    
    # Optional: Prevent transmission if the payload is empty
    if not payload:
        return
        
    for receiver in robot_list:
        # Do not send the data to the broadcaster itself
        if receiver.robot_id != broadcaster.robot_id:
            
            # 2. The receiver parses the incoming data and updates its internal state
            receiver.receive_payload(payload)