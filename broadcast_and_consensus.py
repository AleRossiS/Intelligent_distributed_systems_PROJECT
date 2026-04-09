import config as cfg
import numpy as np

def broadcast_data(broadcaster, robot_list):
    """
    Simulates a network broadcast. The transmitting robot packages its current 
    knowledge and sends it as a payload to all other robots in the system.
    """

    TX_RANGE = 200.0

    # 1. The broadcaster prepares its data packet (dictionary)
    payload = broadcaster.generate_payload()
    
    # Optional: Prevent transmission if the payload is empty
    if not payload:
        return
    
    broadcaster_pos = np.array(broadcaster.state[:2])
        
    for receiver in robot_list:
        # Do not send the data to the broadcaster itself
        if receiver.robot_id != broadcaster.robot_id:
            
            # Check if the receiver is within the transmission range
            receiver_pos = np.array(receiver.state[:2])
            distance = np.linalg.norm(broadcaster_pos - receiver_pos)
            
            if distance <= TX_RANGE:

                # 2. The receiver parses the incoming data and updates its internal state
                receiver.receive_payload(payload)