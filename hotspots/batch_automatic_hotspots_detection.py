"""
General idea here: 
Given the 4 views, we generated a 360° video, 20sec length. 
Idea here is to set N hotspots on the any of the four views, and track them in the 360° video.
We're not going to use the .mp4 output here, only 180 views extracted from the 360° video.
"""

import torch
import sys
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from typing import Tuple
import glob
import os 
import cv2 
from PIL import Image, ImageChops
import numpy as np 
from tqdm import tqdm 
import time 

TEXT_PROMPT_LP = "license plate"
TEXT_PROMPT_DOOR_HANDLES = "door handles"
TEXT_PROMPT_WHEELS = "wheels"

GLOBAL_COUNTER = 1 

from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint

from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device


torch.autocast('cuda',dtype=torch.float16).__enter__()
torch.inference_mode().__enter__()
torch.is_inference_mode_enabled()

def create_empty_datapoint():
    return Datapoint(find_queries = [], images = [])

class Hotspot:
    """Represents a single hotspot to track across images."""
    
    def __init__(self, hotspot_id: int, image_name: str, x: float, y: float, color: Tuple[int, int, int]):
        self.id = hotspot_id
        self.image_name = image_name
        self.init_x = x
        self.init_y = y
        self.color = color
        self.active = False
        self.curr_x = x
        self.curr_y = y
    
    # Activate/deactivate and update setter for a hotspot. 
    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False
    
class SAMHotspots:
    def __init__(self, scene_dir: str,batch_process:bool = False, device: str = "cuda"):

        # Load S3 model. 
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

        # Load the scene. 
        self.scene_dir = scene_dir
        self.video = os.path.join(scene_dir,'videos', "360_final.mp4")

        if batch_process:
            self.datapoint1 = create_empty_datapoint()
            self.set_transform()
            self.set_postprocessor()

    def set_transform(self):
        from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
        self.transform_batch = ComposeAPI([RandomResizeAPI(sizes = 1008,max_size=1008,square=True,consistent_transform=True),
                                    ToTensorAPI(), 
                                    NormalizeAPI(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])])

    def set_postprocessor(self):
        from sam3.eval.postprocessors import PostProcessImage
        self.postprocessor = PostProcessImage(max_dets_per_img = 3,
                                            iou_type = "segm",
                                            use_original_sizes_box = True,
                                            use_original_sizes_mask=True,
                                            convert_mask_to_rle = False,
                                            detection_threshold = 0.7,
                                            to_cpu = False)

    @staticmethod   
    def set_image(datapoint,pil_image):
        w,h = pil_image.size
        datapoint.images = [SAMImage(data = pil_image,
                                    objects = [],
                                    size = [h,w])]
    @staticmethod
    def add_text_prompt(datapoint: Datapoint, text_query: str):
        global GLOBAL_COUNTER

        assert len(datapoint.images) ==1, "Set the image first"
        w,h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(query_text = text_query,
            image_id = 0,
            object_ids_output = [],
            is_exhaustive = [],
            query_processing_order = 0,
            inference_metadata = InferenceMetadata(coco_image_id = GLOBAL_COUNTER,
                                                    original_image_id = GLOBAL_COUNTER,
                                                    original_category_id = 1,
                                                    original_size= [w,h],
                                                    object_id = 0,
                                                    frame_index = 0,
                                                    )
                                )
        )   
        GLOBAL_COUNTER += 1
        return GLOBAL_COUNTER - 1

    def reset(self):
        self.datapoint1 = create_empty_datapoint()
       
   
    def prepare_batch_inference(self,frame:Image.Image):

        SAMHotspots.set_image(self.datapoint1,frame)

        self.id1 = SAMHotspots.add_text_prompt(self.datapoint1,TEXT_PROMPT_LP)
        self.id2 = SAMHotspots.add_text_prompt(self.datapoint1,TEXT_PROMPT_DOOR_HANDLES)
        self.id3 = SAMHotspots.add_text_prompt(self.datapoint1,TEXT_PROMPT_WHEELS)

        self.datapoint1 = self.transform_batch(self.datapoint1) 

    def make_batch(self,datapoint):
        batch =  collate([datapoint],dict_key='dummy')['dummy']
        batch = copy_data_to_device(batch,device = "cuda")
        return batch

    def batch_inference(self):

        cap = cv2.VideoCapture(self.video)
       
        # Get video properties for the output writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        output_filename = os.path.join(self.scene_dir, 'videos', "_batched.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        mask_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # Process the video frame by frame, get a progress bar.
        for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret,frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = frame_bgr[:, :, ::-1]
            pil_image = Image.fromarray(frame_rgb)
           
            self.prepare_batch_inference(pil_image)
            batch = self.make_batch(self.datapoint1)   

            with torch.no_grad():
                output = self.model(batch)
                processed_output = self.postprocessor.process_results(output, batch.find_metadatas)
            
            try:
                mask_lp = processed_output[self.id1]['masks'][0].squeeze(0).cpu()
                mask_image_lp = Image.fromarray(mask_lp.numpy().astype(np.uint8)*255.).convert('RGB')
            except IndexError as e:
                print('No LP mask predicted for frame', i)
               
                mask_image_lp = Image.fromarray(np.zeros((height, width), dtype=np.uint8)).convert('RGB')   
            try:
                mask_handles = processed_output[self.id2]['masks'][0].squeeze(0).cpu()  # It might only consider the first 'door handle' results.
                mask_image_handles = Image.fromarray(mask_handles.numpy().astype(np.uint8)*255.).convert('RGB')
            except IndexError as e:
                print('No handles mask predicted for frame', i)
                mask_image_handles = Image.fromarray(np.zeros((height, width), dtype=np.uint8)).convert('RGB')

            try:
                mask_wheels_final = Image.fromarray(np.zeros((height, width), dtype=np.uint8)).convert('RGB')

                #We need to loop over all the wheels masks and merge them.
                for i in range(len(processed_output[self.id3]['masks'])):
                
                    mask_wheels = processed_output[self.id3]['masks'][i].squeeze(0).cpu()  # It might only consider the first 'door handle' results.
                    mask_wheels_final = ImageChops.add(mask_wheels_final,Image.fromarray(mask_wheels.numpy().astype(np.uint8)*255.).convert('RGB'))
                
            except IndexError as e:
                print('No wheels mask predicted for frame', i)

            mask_image = ImageChops.add(mask_image_lp,mask_image_handles)
            mask_image = ImageChops.add(mask_image,mask_wheels_final)

            blended_res_image = Image.blend(pil_image, mask_image, alpha=0.5)

            # Write the frame to the video writer
            mask_writer.write(np.asarray(blended_res_image)[:, :, ::-1]) 

            self.reset()    
        cap.release()
        mask_writer.release()

        

    def vanilla_single_tracking(self):
        cap = cv2.VideoCapture(self.video)
       
        # Get video properties for the output writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        output_filename = os.path.join(self.scene_dir, 'videos', "360_final_single_.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        mask_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # Process the video frame by frame, get a progress bar.
        for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret,frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = frame_bgr[:, :, ::-1]
            pil_image = Image.fromarray(frame_rgb)
            state = self.processor.set_image(pil_image)

            output_dict = self.processor.set_text_prompt(state=state, prompt=TEXT_PROMPT_LP)

            try:
                masks = output_dict['masks'][0].squeeze(0).cpu()
                mask_image = Image.fromarray(masks.numpy().astype(np.uint8)*255.).convert('RGB')
                blended_res_image = Image.blend(pil_image, mask_image, alpha=0.5)

            except IndexError as e:
                print('No mask predicted for frame', i)
               # Blend with a black mask.
                mask_image = Image.fromarray(np.zeros((height, width), dtype=np.uint8)).convert('RGB')
                blended_res_image = Image.blend(pil_image, mask_image, alpha=0.5)         

            # Write the frame to the video writer
            mask_writer.write(np.asarray(blended_res_image)[:, :, ::-1]) 

        cap.release()
        mask_writer.release()

if __name__=='__main__':
    
    sam_hotspots = SAMHotspots(scene_dir="/home/gaetan/data/nextgen360/Renault/",batch_process=True)
    sam_hotspots.batch_inference()


