import logging
from transformers import Sam2Processor, Sam2Model, Sam3Processor, Sam3Model
from accelerate import Accelerator
import torch
import numpy as np
import re

TASK_TYPE = {
    "open":     [1,0,0],
    "close":    [0,1,0],
    "relocate": [0,0,1],
    "move":     [0,0,1],
    "put":      [0,0,1],
    "pick":     [0,0,1],
    "place":    [0,0,1],
    "remove":   [0,0,1],
    "grasp":    [0,0,1],
    "take":     [0,0,1],
    "lift":     [0,0,1],
    "turn":     [0,0,1],
    "rotate":   [0,0,1],
    "push":     [0,0,1],
    "pull":     [0,0,1],
}

RED_FLAG = [
    "and",
    "or",
    "then",
]

class GoalOracle:
    """
    Generates standardized 38D goal embeddings for robot manipulation tasks.

    Goal Vector Layout (38D):
    | Indices | Component                              |
    |---------|----------------------------------------|
    | 0-2     | Task type 1-hot [Open, Close, Relocate]|
    | 3-6     | Initial Pos (2D BB): [x, y, W, H]      |
    | 7-10    | Target Pos (2D BB): [x, y, W, H]       |
    | 11-13   | Object color [R, G, B] Norm [0, 1]     |
    """

    def __init__(self, device: str | torch.device = "cpu"):
        device = Accelerator().device
        self.device = device
        self.SAM2 = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
        self.SAM2_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        self.SAM3 = Sam3Model.from_pretrained("facebook/sam3").to(device)
        self.SAM3_processor = Sam3Processor.from_pretrained("facebook/sam3")
        
        logging.info("GoalOracle initialization complete.")
        logging.info(f"Initialized GoalOracle on device: {device}")


    def encode_goal(
        self,
        init_obj_coordinate: np.ndarray | None,
        final_obj_coordinate: np.ndarray | None,
        img: np.ndarray | None = None,
        instruction: str | None = None,
    ) -> torch.Tensor | None:
        """
        Construct standardized 14D goal vector.

        Args:
            init_obj_coordinate: 2D initial object coordinate [x, y]
            final_obj_coordinate: 2D final object coordinate [x, y]
            img: Optional RGB image (H, W, 3) for visual context (not used in this version)
            instruction: Optional natural language instruction for property extraction

        Returns:
            (14,) float32 tensor
        """
        if instruction is not None and any(flag in instruction.lower() for flag in RED_FLAG):
            return torch.zeros(14, dtype=torch.float32).to(self.device)  # Return zero vector if red flags are present
        # Task type encoding (3D)
        task_type_vec = self._encode_task_type(instruction)

        if task_type_vec is None:
            logging.warning(f"No task type keywords found in instruction: '{instruction}'")
            return None
        # Object position encoding (8D)
        init_pos_vec = self._encode_position(task_type_vec, init_obj_coordinate, img, instruction)
        if np.array_equal(task_type_vec, np.array([0, 0, 1], dtype=np.float32)):
            final_pos_vec = self._encode_position(task_type_vec, final_obj_coordinate, img, instruction)
        else:
            final_pos_vec = init_pos_vec  # For relocate tasks, initial and final positions are the same            

        # Object property encoding (3D)
        prop_vec = self._encode_properties(img, init_pos_vec)

        # Concatenate all components into a single goal vector
        if init_pos_vec is None or final_pos_vec is None or prop_vec is None:
            logging.warning(f"Failed to encode one of the components for instruction: '{instruction}'")
            return None
        goal_vector = np.concatenate([task_type_vec, init_pos_vec, final_pos_vec, prop_vec])
        return torch.tensor(goal_vector, dtype=torch.float32).to(self.device)
    
    def _encode_task_type(self, instruction: str | None) -> np.ndarray | None:
        if instruction is None:
            return None
        
        instruction = instruction.lower()
        for keyword, vec in TASK_TYPE.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', instruction):
                return np.array(vec, dtype=np.float32)
        return None
    
    def _encode_position(self, task_type: np.ndarray, coordinate: np.ndarray | None, image: np.ndarray | None, instruction: str | None) -> np.ndarray | None:
        if np.array_equal(task_type, np.array([0, 0, 1], dtype=np.float32)):
            coordinate = [[[coordinate]]]
            input_labels = [[[1]]]
            inputs = self.SAM2_processor(images=image, input_points=coordinate, input_labels=input_labels, return_tensors="pt").to(self.SAM2.device)

            with torch.no_grad():
                outputs = self.SAM2(**inputs)
            
            results = self.SAM2_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
            try:
                masks = results[0][0].cpu().numpy()
            except IndexError:
                logging.warning(f"SAM2 failed to find a mask for coordinate: {coordinate}")
                return None
            
        else:
            # for non-relocate tasks use florence to extract position from instruction and image
            if instruction is None or image is None:
                return None

            # Clean the instruction to focus only on the object
            target_desc = instruction.lower()
            for action in TASK_TYPE.keys():
                target_desc = target_desc.replace(action, "")

            inputs = self.SAM3_processor(images=image, text=target_desc, return_tensors="pt").to(self.SAM3.device)

            with torch.no_grad():
                outputs = self.SAM3(**inputs)
            results = self.SAM3_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            try:
                masks = results["masks"][0].cpu().numpy()
            except IndexError:
                logging.warning(f"SAM3 failed to find a mask for instruction: '{instruction}'")
                return None

        
        # Now np.where will correctly return 2 values: ys and xs
        ys, xs = np.where(masks > 0)
        ys, xs = np.where(masks > 0)
        if len(xs) == 0 or len(ys) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        return np.array([x_center, y_center, width, height], dtype=np.float32)

    def _encode_properties(self, image: np.ndarray | None, bounding_box: np.ndarray | None) -> np.ndarray | None:
        if image is None or bounding_box is None:
            return None

        if bounding_box is None:
            return None

        x, y, W, H = bounding_box
        y1, y2 = int(y), int(y + H)
        x1, x2 = int(x), int(x + W)
        
        crop = image[y1:y2, x1:x2]
        
        # Prevent crash if box is out of bounds
        if crop.size == 0:
            return None
            
        avg_color = crop.mean(axis=(0, 1))
        
        # Normalize to [0, 1] if your previous code expected it
        return (avg_color / 255.0).astype(np.float32)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    oracle = GoalOracle("cuda" if torch.cuda.is_available() else "cpu")
    init_coord1 = np.array([104, 166])
    final_coord1 = np.array([98, 150])
    instruction1 = "Move the Green block inside the basket"
    from PIL import Image
    
    # Open the image
    img1 = Image.open('images/GoalOracleTest1.jpg')

    # Convert to numpy array
    img_array1 = np.array(img1)
    goal_vector1 = oracle.encode_goal(init_coord1, final_coord1, img=img_array1, instruction=instruction1)
    logging.info(f"Encoded Goal Vector: {goal_vector1.cpu().numpy()}")
    logging.info(f"Task Type (One-hot): {goal_vector1[:3].cpu().numpy()}")
    logging.info(f"Initial Position (x, y, W, H): {goal_vector1[3:7].cpu().numpy()}")
    logging.info(f"Target Position (x, y, W, H): {goal_vector1[7:11].cpu().numpy()}")
    logging.info(f"Object Color (R, G, B): {goal_vector1[11:14].cpu().numpy()}")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_array1)
    
    
    cx, cy, w, h = goal_vector1[3:7].cpu().numpy()  # Initial position
    # Convert center-format to top-left for matplotlib
    rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=2, edgecolor="blue", facecolor='none', label="Initial")
    ax.add_patch(rect)
    ax.scatter(cx, cy, color="blue", s=40)
    cx, cy, w, h = goal_vector1[7:11].cpu().numpy()  # Final position
    rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=2, edgecolor="red", facecolor='none', label="Final")
    ax.scatter(cx, cy, color="red", s=40)
    ax.add_patch(rect)

    #display color
    r, g, b = goal_vector1[11:14].cpu().numpy()
    ax.add_patch(patches.Rectangle((0, 0), 50, 50, linewidth=2, edgecolor="black", facecolor=(r, g, b), label="Object Color"))

    plt.legend()
    plt.title(f"Goal Oracle Detection\nInstruction: {instruction1}")
    plt.show()

    #--- Test with Open flag in instruction
    instruction2 = "Open the bottom white drawer on the left side of the cabinet"
    
    # Open the image
    img2 = Image.open('images/GoalOracleTest2.jpg')
    init_coord2 = None
    final_coord2 = None

    # Convert to numpy array
    img_array2 = np.array(img2)
    goal_vector2 = oracle.encode_goal(init_coord2, final_coord2, img=img_array2, instruction=instruction2)
    logging.info(f"Encoded Goal Vector: {goal_vector2.cpu().numpy()}")
    logging.info(f"Task Type (One-hot): {goal_vector2[:3].cpu().numpy()}")
    logging.info(f"Initial Position (x, y, W, H): {goal_vector2[3:7].cpu().numpy()}")
    logging.info(f"Target Position (x, y, W, H): {goal_vector2[7:11].cpu().numpy()}")
    logging.info(f"Object Color (R, G, B): {goal_vector2[11:14].cpu().numpy()}")

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_array2)
    
    
    cx, cy, w, h = goal_vector2[3:7].cpu().numpy()  # Initial position
    # Convert center-format to top-left for matplotlib
    rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=2, edgecolor="blue", facecolor='none', label="Initial")
    ax.add_patch(rect)
    ax.scatter(cx, cy, color="blue", s=40)
    cx, cy, w, h = goal_vector2[7:11].cpu().numpy()  # Final position
    rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=2, edgecolor="red", facecolor='none', label="Final")
    ax.scatter(cx, cy, color="red", s=40)
    ax.add_patch(rect)

    #display color
    r, g, b = goal_vector2[11:14].cpu().numpy()
    ax.add_patch(patches.Rectangle((0, 0), 50, 50, linewidth=2, edgecolor="black", facecolor=(r, g, b), label="Object Color"))

    plt.legend()
    plt.title(f"Goal Oracle Detection\nInstruction: {instruction2}")
    plt.show()
