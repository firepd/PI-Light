from typing import List, Tuple
from utilities.utils import list_with_unique_element


class RoadLink:
    def __init__(self, roadlink_dict, intersection):
        self.intersection = intersection

        self.startroad_id = roadlink_dict['startRoad']  # str 入射路的名字
        self.endroad_id = roadlink_dict['endRoad']  # str 出射路的名字

        self.n_lanelink_id = []  # type: List[Tuple[str, str]]  # 车道组合, 入射车道是确定的, 出射车道不确定
        self.n_startlane_id = []  # type: List[str]  # 正常来说只有一个元素
        self.n_endlane_id = []
        for lanelink_dict in roadlink_dict['laneLinks']:
            startlane_id = '{}_{}'.format(self.startroad_id, lanelink_dict['startLaneIndex'])
            endlane_id = '{}_{}'.format(self.endroad_id, lanelink_dict['endLaneIndex'])
            self.n_lanelink_id.append((startlane_id, endlane_id))
            self.n_startlane_id.append(startlane_id)
            self.n_endlane_id.append(endlane_id)
        self.n_lanelink_id = list_with_unique_element(self.n_lanelink_id)  # 去重
        self.n_startlane_id = list_with_unique_element(self.n_startlane_id)
        self.n_endlane_id = list_with_unique_element(self.n_endlane_id)

    def __str__(self):
        return str({'startroad_id': self.startroad_id,
                    'endroad_id': self.endroad_id,
                    'n_lanelink_id': self.n_lanelink_id})
