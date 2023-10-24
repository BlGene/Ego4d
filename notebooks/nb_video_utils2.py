import numpy as np
def frame_and_name2pos(frame, name):
    bboxes = frame['boxes']
    # TODO(max): this ignores the cases where we have two objects.
    object_names = [x['object_type'] for x in bboxes]
    occurences = sum(x == name for x in object_names)
    if occurences == 0:
        return np.array((np.nan, np.nan))
    if occurences > 1:
        print("Warning: two instances of object", object_names, name)
        return np.array((np.nan, np.nan))

    bbox = bboxes[object_names.index(name)]['bbox']
    center_x = bbox['x'] + bbox['width']/2.
    center_y = bbox['y'] + bbox['height']/2.
    return np.array((center_x, center_y))

def assign_rl(frame: list) -> str:
    bboxes = frame['boxes']
    object_names = [x['object_type'] for x in bboxes]
    if 'object_of_change' not in object_names:
        return 'unknown'
    if not ('right_hand' in object_names or 'left_hand' in object_names):
        return 'unknown'
    pos_r = frame_and_name2pos(frame, 'right_hand')
    pos_l = frame_and_name2pos(frame, 'left_hand')
    pos_o = frame_and_name2pos(frame, 'object_of_change')
    d_r = np.nan_to_num(np.linalg.norm(pos_o-pos_r), nan=np.inf)
    d_l = np.nan_to_num(np.linalg.norm(pos_o-pos_l), nan=np.inf)
    if d_r == np.inf and d_l == np.inf:
        return "unknown"
    elif d_r < d_l:
        return "right_hand"
    else:
        return "left_hand"

def frames2hand(frames:list):
    frame_types = [fr['frame_type'] for fr in frames]
    if 'pnr_frame' in frame_types:
        frame_index = frame_types.index('pnr_frame')
    elif 'contact_frame' in frame_types:
        frame_index = frame_types.index('contact_frame')
    elif 'post_frame' in frame_types:
        frame_index = frame_types.index('post_frame')
    else:
        return None
    hand = assign_rl(frames[frame_index])
    return hand

def box2center(bbox):
    center_x = bbox['x'] + bbox['width']/2.
    center_y = bbox['y'] + bbox['height']/2.
    return np.array((center_x, center_y))


def get_frame_centers(frames3: list):
    centers = dict(left_hand=np.zeros((len(frames3), 2))+np.nan,
                   right_hand=np.zeros((len(frames3), 2))+np.nan,
                   object_of_change=np.zeros((len(frames3), 2))+np.nan,
                   names=[fr['frame_type'] for fr in frames3])
    for i, frame in enumerate(frames3):
        for box in frame['boxes']:
            if box['object_type'] not in centers:
                continue
            centers[box['object_type']][i] = box2center(box['bbox'])
    return centers

def plot_frames(frame_paths, frames):
    for frame_path, frame_data in zip(frame_paths, frames):
        dest = plot_dir / f'{video["video_uid"]}_{frame_data["frame_number"]:08d}.jpg'
        shutil.copyfile(frame_path, dest)
        #display(NBImage(filename=frame_path))
        #print(f"↑ a:{a} {frame_data['frame_type']} frame num: {frame_data['frame_number']}  -> {dest}\n")
