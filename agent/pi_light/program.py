import numpy as np
from itertools import combinations, product
from copy import deepcopy
import random


max_block_breadth = 3
max_depth = 2
max_num_if = 2  # Up to 2 if and if-else
max_weight_param = 2
feat_list = ['inlane_2_num_vehicle', 'inlane_2_num_waiting_vehicle', 'outlane_2_num_vehicle', 'inlane_2_vehicle_dist', 'outlane_2_vehicle_dist']
vec_feats = ['inlane_2_vehicle_dist', 'outlane_2_vehicle_dist']
scalar_feats = list(set(feat_list) - set(vec_feats))


def code_indent(code: str, depth: int):
    if depth == 0:
        return code
    indentation = depth * 4
    code_lines = code.strip().split('\n')
    new = []
    for i in code_lines:
        new.append(indentation * ' ' + i)
    return '\n'.join(new)


class Condition:
    def __init__(self, feat_name: str, compare_op: str):
        self.feat_name = feat_name
        self.compare_op = compare_op
        self.value = None

    def flip(self):
        pairs = {'>=':'<', '>':'<=', '<=':'>', '<':'>='}
        flipped = pairs[self.compare_op]
        return Condition(self.feat_name, flipped)

    def to_str(self):
        placeholder = f'$cond:{self.feat_name}$'
        return Tool.locate(self.feat_name) + self.compare_op + placeholder


class Instruction:
    def __init__(self, feat_name: str):
        self.feat_name = feat_name
        self.inlane = True if 'inlane' in feat_name else False
        self.feat_is_vec = feat_name in vec_feats
        self.num_weight = 0
        self.num_threshold = 1 if self.feat_is_vec else 0

    def can_add_constant(self):
        return True if self.num_weight == 0 else False

    def add_constant(self):
        self.num_weight += 1

    def get_python_code(self, depth) -> object:
        left_code = 'value[0] += ' if self.inlane else 'value[0] -= '
        feat_index = Tool.locate(self.feat_name)
        if self.feat_is_vec:
            placeholder = f'$thresh:{self.feat_name}$'
            indicator = feat_index + '<' + placeholder
            right_code = Tool.dot_sum(indicator)
        else:
            right_code = feat_index

        if self.num_weight == 1:
            right_code += f' * $weight:{self.feat_name}$'

        return code_indent(left_code + right_code, depth)

    def length(self):
        return 1


class If:
    def __init__(self, condition: Condition):
        self.condition = condition
        self.if_sub_program = Block()

    def get_python_code(self, depth):
        if_code = 'if ' + self.condition.to_str() + ':'
        sub_code = self.if_sub_program.get_python_code(depth + 1)
        codes = code_indent(if_code, depth) + '\n' + sub_code
        return codes

    def length(self):
        return 1 + self.if_sub_program.length()

    def height(self):
        return self.if_sub_program.height() + 1

    def num_weight(self):
        return self.if_sub_program.num_weight()

    def num_condition(self):
        return self.if_sub_program.num_condition() + 1


class IfElse:
    def __init__(self, condition: Condition):
        self.condition = condition
        self.if_sub_program = Block()
        self.else_sub_program = Block()

    def get_python_code(self, depth):
        if_code = 'if ' + self.condition.to_str() + ':'
        if_code = code_indent(if_code, depth)
        sub_code1 = self.if_sub_program.get_python_code(depth + 1)
        else_code = code_indent('else:', depth)
        sub_code2 = self.else_sub_program.get_python_code(depth + 1)
        return '\n'.join([if_code, sub_code1, else_code, sub_code2])

    def length(self):
        return self.if_sub_program.length() + self.if_sub_program.length() + 2

    def height(self):
        higher = max(self.if_sub_program.height(), self.else_sub_program.height())
        return higher + 1

    def num_weight(self):
        return self.if_sub_program.num_weight() + self.else_sub_program.num_weight()

    def num_condition(self):
        return self.if_sub_program.num_condition() + self.else_sub_program.num_condition() + 1


class Block:
    def __init__(self):
        self.data = []

    def get_python_code(self, depth):
        codes = [component.get_python_code(depth) for component in self.data]
        code = '\n'.join(codes)  # If codes is [], you can get ''
        return code

    def add(self, component):
        self.data.append(component)

    def can_add_component(self):
        return True if len(self.data) < max_block_breadth else False

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):  # block[0] = 1
        self.data[index] = value

    def length(self):  # overall code length
        lens = [component.length() for component in self.data]
        return sum(lens)

    def height(self):
        heights = [i.height() for i in self.data if type(i) != Instruction]
        if len(heights) == 0:
            return 0
        return max(heights)

    def can_build_higher_bug(self):  # Even if the maximum height is reached, other low heights can be expanded
        return True if self.height() < max_depth else False

    def num_weight(self):
        num = 0
        for i in self.data:
            if type(i) == Instruction:
                num += i.num_weight
            else:
                num += i.num_weight()
        return num

    def num_condition(self):
        return sum([i.num_condition() for i in self.data if type(i) != Instruction])

    def complexity(self):
        return self.length() + self.num_condition()


class Tool:
    @classmethod
    def locate(cls, feat: str):
        return feat + '[index]'
    @classmethod
    def dot_sum(cls, code: str):
        return '(' + code + ').sum()'
    @classmethod
    def dot_max(cls, code: str):
        return '(' + code + ').max()'


class Program:
    def __init__(self, inlane):
        self.options = ['if', 'if-else']
        self.block = Block()
        if inlane:
            self.own_feat_list = [i for i in feat_list if 'inlane' in i]
        else:
            self.own_feat_list = [i for i in feat_list if 'outlane' in i]

    def get_valid_expansions(self):
        to_return = []
        # for i, component in enumerate(self.block):
        #     if type(component) is Instruction and component.can_add_constant():
        #         new_program = deepcopy(self)
        #         new_program.block[i].add_constant()
        #         to_return.append(new_program)

        # Find all instructions without parameters and add a parameter to him
        instruct_pos = []
        if self.block.num_weight() < max_weight_param:
            self.findall_ins_no_weight(self.block, instruct_pos, [])
        for pos in instruct_pos:
            new_program = deepcopy(self)
            instruction = new_program.navigate_to_pos(pos)
            instruction.add_constant()
            to_return.append(new_program)

        # Find all the blocks that are not full, and put an instruction in the end
        wait_add_block_pos = []
        self.findall_block_wait_expand(self.block, wait_add_block_pos, [])
        for pos in wait_add_block_pos:
            block = self.navigate_to_pos(pos)
            feats = self.get_valid_instruction_feat(block)
            for feat in feats:
                new_program = deepcopy(self)
                block = new_program.navigate_to_pos(pos)
                instruction = Instruction(feat)
                block.add(instruction)
                to_return.append(new_program)

        # Find single instructions and turn them into if-else
        ins_path = []
        if self.block.num_condition() < max_num_if:
            self.findall_ins_expand2_if(self.block, ins_path, [])
        for path in ins_path:
            pos, idx = path[:-1], path[-1]
            random_conditions = self.get_random_conditions()
            for r in random_conditions:
                new_program = deepcopy(self)
                block = new_program.navigate_to_pos(pos)
                if_comp = If(r)
                if_comp.if_sub_program.add(block[idx])
                block[idx] = if_comp
                to_return.append(new_program)

        # if-else module
        for path in ins_path:
            pos, idx = path[:-1], path[-1]
            random_conditions = self.get_random_conditions()
            random_instruction = self.get_random_instruction()
            for r in random_conditions:
                for ins in random_instruction:
                    new_program = deepcopy(self)
                    block = new_program.navigate_to_pos(pos)
                    if_comp = IfElse(r)
                    if_comp.if_sub_program.add(block[idx])
                    if_comp.else_sub_program.add(ins)
                    block[idx] = if_comp
                    to_return.append(new_program)

        return to_return

    def findall_ins_no_weight(self, block, to_append: list, ancestor_sequence: list):
        for i, compon in enumerate(block):
            if type(compon) is Instruction and compon.can_add_constant():
                to_append.append(ancestor_sequence + [i])
            elif type(compon) is If:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_ins_no_weight(compon.if_sub_program, to_append, new_ancestor)
            elif type(compon) is IfElse:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_ins_no_weight(compon.if_sub_program, to_append, new_ancestor)
                new_ancestor = ancestor_sequence + [i] + ['else_sub_program']
                self.findall_ins_no_weight(compon.else_sub_program, to_append, new_ancestor)
            else:
                pass

    def findall_block_wait_expand(self, block, to_append: list, ancestor_sequence: list):
        if block.can_add_component() and len(self.get_valid_instruction_feat(block)) > 0:
            to_append.append(ancestor_sequence)
        for i, compon in enumerate(block):
            if type(compon) is If:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_block_wait_expand(compon.if_sub_program, to_append, new_ancestor)
            elif type(compon) is IfElse:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_block_wait_expand(compon.if_sub_program, to_append, new_ancestor)
                new_ancestor = ancestor_sequence + [i] + ['else_sub_program']
                self.findall_block_wait_expand(compon.else_sub_program, to_append, new_ancestor)

    def get_valid_instruction_feat(self, block):
        ins_feat_name = [compon.feat_name for compon in block if type(compon) is Instruction]
        return set(self.own_feat_list) - set(ins_feat_name)

    def findall_ins_expand2_if(self, block, to_append: list, ancestor_sequence: list):
        if len(ancestor_sequence) >= max_depth*2:  # Don't let him go deeper than 2
            return
        flag = False
        for i, compon in enumerate(block):
            if type(compon) is Instruction and flag is False:
                flag = True
                to_append.append(ancestor_sequence + [i])
            if type(compon) is If:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_ins_expand2_if(compon.if_sub_program, to_append, new_ancestor)
            elif type(compon) is IfElse:
                new_ancestor = ancestor_sequence + [i] + ['if_sub_program']
                self.findall_ins_expand2_if(compon.if_sub_program, to_append, new_ancestor)
                new_ancestor = ancestor_sequence + [i] + ['else_sub_program']
                self.findall_ins_expand2_if(compon.else_sub_program, to_append, new_ancestor)

    def navigate_to_pos(self, path):
        if len(path) == 0:
            return self.block
        cur_obj = self.block[path[0]]
        for i in path[1:]:
            if type(i) is str:
                cur_obj = getattr(cur_obj, i)
            elif type(i) is int:
                cur_obj = cur_obj[i]
            else:
                assert 0, i
        return cur_obj

    def get_random_conditions(self, if_else=False):
        to_return = []
        for feat in self.own_feat_list:
            if feat in vec_feats:
                continue
            for op in ['<=', '>']:
                if op == '>' and if_else:
                    continue
                cond = Condition(feat, op)
                to_return.append(cond)
        return to_return

    def get_random_instruction(self):
        to_return = []
        for feat in self.own_feat_list:
            instruct = Instruction(feat)
            to_return.append(instruct)
        return to_return

    def length(self):
        return self.block.length()

    def get_python_code(self):
        return self.block.get_python_code(0)


class Bale:
    def __init__(self):
        self.inlane_program = Program(inlane=True)
        self.outlane_program = Program(inlane=False)
        self.action_fingerprint = None
        self.in_code = None
        self.out_code = None

    def output_code(self):
        if self.in_code is None:
            self.in_code = self.inlane_program.get_python_code()
            self.out_code = self.outlane_program.get_python_code()
        return self.in_code, self.out_code

    def replace_code(self, inlane_code, outlane_code):
        self.in_code, self.out_code = inlane_code, outlane_code

    @staticmethod
    def get_start_programs():
        temp = Program(inlane=True)
        to_return = []
        for feat in temp.own_feat_list:
            instruct = Instruction(feat)
            new_bale = Bale()
            new_bale.inlane_program.block.add(instruct)
            to_return.append(new_bale)
        return to_return

    def get_valid_expansions(self):
        high_priority = []
        if self.outlane_program.length() == 0:
            temp = Program(inlane=False)
            for feat in temp.own_feat_list:
                instruct = Instruction(feat)
                new_bale = self.copy()
                new_bale.outlane_program.block.add(instruct)
                high_priority.append(new_bale)
            return high_priority

        to_return = []
        inlane_expansions = self.inlane_program.get_valid_expansions()
        for in_e in inlane_expansions:
            new_bale = self.copy(inlane=False)
            new_bale.inlane_program = in_e
            to_return.append(new_bale)

        outlane_expansions = self.outlane_program.get_valid_expansions()
        for out_e in outlane_expansions:
            new_bale = self.copy(outlane=False)
            new_bale.outlane_program = out_e
            to_return.append(new_bale)

        return to_return

    def copy(self, inlane=True, outlane=True):
        new_bale = Bale()
        if inlane:
            new_bale.inlane_program = deepcopy(self.inlane_program)
        if outlane:
            new_bale.outlane_program = deepcopy(self.outlane_program)
        return new_bale

    def get_complexity(self):
        return self.inlane_program.block.complexity() + self.outlane_program.block.complexity()



