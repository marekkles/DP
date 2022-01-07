from typing import Optional
import copy
from PIL import Image,ImageDraw
import os

class IrisImage(object):
    def __init__(self, header: Optional[list] = None, param_list: Optional[list] = None, dataset_root: Optional[str] = None) -> None:
        super().__init__()
        self.null()
        if header and param_list and dataset_root:
            self.update(header, param_list, dataset_root)
    def null(self):
        self.param_map = {}
        self.dataset_root = None
        self.pos_x = None
        self.pos_y = None
        self.radius_0 = None
        self.radius_1 = None
        self.image_data = None
        self.image_mask = None
    def update_params(self, header: list, param_list: list):
        for key, val in zip(header, param_list):
            self.param_map[key] = val
    def update_circles(self):
        assert (
            'pos_x' in self.param_map and
            'pos_y' in self.param_map and
            'radius_0' in self.param_map and
            'radius_1' in self.param_map
        ), 'Parameter list does not contain information about circles in the header'

        self.pos_x    = float(self.param_map['pos_x'])
        self.pos_y    = float(self.param_map['pos_y'])
        self.radius_0 = float(self.param_map['radius_0'])
        self.radius_1 = float(self.param_map['radius_1'])
    def update_mask(self):
        assert (
            self.pos_x and 
            self.pos_y and 
            self.radius_0 and 
            self.radius_1
        ), 'Position of circles has not been set/updated'
        
        self.image_mask = Image.new(mode='1', size=self.image_data.size, color=0)
        image_draw = ImageDraw.Draw(self.image_mask)
        image_draw.ellipse((
            self.pos_x-self.radius_1,
            self.pos_y-self.radius_1,
            self.pos_x+self.radius_1,
            self.pos_y+self.radius_1),
        fill=1)
        image_draw.ellipse((
            self.pos_x-self.radius_0,
            self.pos_y-self.radius_0,
            self.pos_x+self.radius_0,
            self.pos_y+self.radius_0),
        fill=0)
    def reload(self):
        assert self.dataset_root, 'dataset_root is not set' 
        assert 'image_id' in self.param_map, 'Parameter list does not contain image_id in the header'
        
        image_dir = self.param_map['image_id']
        img_path = os.path.join(self.dataset_root, image_dir, 'iris_right.UNKNOWN')
        self.image_data = Image.open(img_path)
    def update(self, header: list, param_list: list, dataset_root: str):
        self.null()
        self.dataset_root = dataset_root
        self.update_params(header, param_list)
        self.update_circles()
        self.reload()
    def get(self):
        assert self.image_data, 'No image is loaded, cannot get a copy'

        return copy.deepcopy(self.image_data) 
    def get_mask(self):
        assert self.image_mask, 'No image is loaded, cannot get a copy'

        return copy.deepcopy(self.image_mask) 
    def crop(self, size: Optional[int] = None):
        assert (
            self.pos_x and 
            self.pos_y and 
            self.radius_0 and 
            self.radius_1
        ), 'Position of circles has not been set/updated'

        self.image_data = self.image_data.crop((
            self.pos_x-self.radius_1,
            self.pos_y-self.radius_1,
            self.pos_x+self.radius_1,
            self.pos_y+self.radius_1
        ))

        self.pos_x = self.radius_1
        self.pos_y = self.radius_1
        
        if size != None:
            assert type(size)== int, 'Size should be a single integer'
            
            factor = (size/2) / self.radius_1 
            self.radius_0 *= factor 
            self.radius_1 *= factor
            self.pos_x = self.radius_1
            self.pos_y = self.radius_1
            
            self.image_data = self.image_data.resize((size,size))
    def paint_circles(self):
        assert (
            self.pos_x and 
            self.pos_y and 
            self.radius_0 and 
            self.radius_1
        ), 'Position of circles has not been set/updated'

        point_radius = 5
        image_draw = ImageDraw.Draw(self.image_data)
        image_draw.ellipse((
            self.pos_x-point_radius,
            self.pos_y-point_radius,
            self.pos_x+point_radius,
            self.pos_y+point_radius),
        fill='red')
        image_draw.ellipse((
            self.pos_x-self.radius_0,
            self.pos_y-self.radius_0,
            self.pos_x+self.radius_0,
            self.pos_y+self.radius_0), 
        outline ='blue', width=point_radius)
        image_draw.ellipse((
            self.pos_x-self.radius_1,
            self.pos_y-self.radius_1,
            self.pos_x+self.radius_1,
            self.pos_y+self.radius_1), 
        outline ='green', width=point_radius)