from dataclasses import dataclass
from jinja2 import Template
import pandas as pd
import pathlib
from typing import Optional, List, Tuple
from model_interface import CustomResponse, prompt_model


ROOT_DIR = pathlib.Path(__file__).parent
PROMPT_DIR = ROOT_DIR / 'prompts'


@dataclass
class Answer:
    question_category: Optional[List[str]]
    user_interaction_id_cat: Optional[int]
    n_input_tokens_cat: Optional[int]
    n_output_tokens_cat: Optional[int]
    used_cache_cat: Optional[bool]
    model_answer: str
    user_interaction_id_answer: Optional[int]
    n_input_tokens_answer: Optional[int]
    n_output_tokens_answer: Optional[int]
    used_cache_answer: Optional[bool]


def command_row_to_text(cmd_row: pd.Series) -> str:
    txt = f"Name: {cmd_row['command']}\n"
    txt += f"Syntax: {cmd_row['syntax']}\n"
    txt += f"Help: {cmd_row['help']}\n"
    txt += f"Permission: {cmd_row['permission']}\n"
    return txt


class Assistance:
    def __init__(self):
        with open(PROMPT_DIR / 'classify_question.txt', 'r') as f:
            self.classify_question_prefix = f.read()

        self.commands_categories = pd.read_csv(ROOT_DIR / 'documents' / 'commands_categories.csv')
        self.all_commands = self._load_all_commands()

        with open(PROMPT_DIR / 'answer_question_system_message.txt', 'r') as f:
            self.answer_question_system_message = f.read()

        with open(PROMPT_DIR / 'answer_question.txt') as f:
            self.answer_question_prompt_template = Template(f.read())

    @staticmethod
    def _load_all_commands() -> pd.DataFrame:
        with open(ROOT_DIR / 'documents' / 'rc_server_commands_all.txt', 'r') as f:
            all_commands_txt = f.read()
        all_commands_list = all_commands_txt.split('\n\n')
        df = []
        for one_command_txt in all_commands_list:
            cmd_name = one_command_txt.split('Name: ')[1].split('\n')[0]
            cmd_syntax = one_command_txt.split('Syntax: ')[1].split('\n')[0]
            cmd_help = one_command_txt.split('Help: ')[1].split('\n')[0]
            cmd_permission = one_command_txt.split('Permission: ')[1].split('\n')[0]
            if 'Plugin: ' in one_command_txt:
                cmd_plugin = one_command_txt.split('Plugin: ')[1].split('\n')[0]
            else:
                cmd_plugin = None
            df.append({'command': cmd_name, 'syntax': cmd_syntax, 'help': cmd_help, 'permission': cmd_permission,
                       'plugin': cmd_plugin})
        df = pd.DataFrame(df)
        return df

    def classify_question(self, user_question: str) -> Tuple[List[str], CustomResponse]:
        prompt = f"{self.classify_question_prefix} \"{user_question}\"\nCategories: "
        messages = [{'role': 'user', 'content': prompt}]
        model_resp = prompt_model(messages=messages)
        model_content = model_resp.model_content
        classification_output = model_content.split(',')
        classification_output = [cl_o.strip().lower().replace('game play', 'gameplay')
                                 for cl_o in classification_output]
        if 'irrelevant' in classification_output:
            return ['irrelevant']
        classification_mapping = {
            'beacons mines and c4s': ['beacons_mines_c4'],
            'general information': ['info'],
            'kick and ban': ['kick_ban'],
            'muting': ['mute'],
            'messages and communication': ['messages'],
            'gameplay commands and player control': ['players_management', 'game_management'],
            'team management': ['team_management'],
            'ranking and statistics': ['ranks'],
            'moderators management': ['moderation_settings'],
            'server management': ['server_settings'],
        }
        category = [classification_mapping.get(cl_o, None) for cl_o in classification_output]
        category = [cat for cat in category if cat is not None]
        category = [x for xs in category for x in xs]  # flatten the list
        if not category:
            category = ['all']
        return category, model_resp

    def get_commands_by_category(self, question_category: List[str]) -> str:
        if 'all' in question_category:
            commands_category_df = self.commands_categories.copy()
        else:
            commands_category_df = self.commands_categories[['command'] + question_category]
        commands_category_df = commands_category_df.dropna(axis=0, thresh=2)
        commands_category_df = commands_category_df.merge(right=self.all_commands, on='command', how='inner')

        commands_category = []
        for _, row in commands_category_df.iterrows():
            commands_category.append(command_row_to_text(row))
        commands_category = '\n'.join(commands_category)
        return commands_category

    def answer_user_question(self, user_question: str, question_category: List[str]) -> CustomResponse:
        commands = self.get_commands_by_category(question_category=question_category)
        prompt = self.answer_question_prompt_template.render(commands=commands, user_question=user_question)
        messages = [{'system': self.answer_question_system_message, 'role': 'user', 'content': prompt}]
        model_resp = prompt_model(messages=messages)
        return model_resp


assistance = Assistance()


def answer_a_question(user_question: str, find_category_first: bool = False) -> Answer:
    if find_category_first:
        question_category, question_category_resp = assistance.classify_question(user_question=user_question)
        user_interaction_id_cat = question_category_resp.user_interaction_id
        n_input_tokens_cat = question_category_resp.n_input_tokens
        n_output_tokens_cat = question_category_resp.n_output_tokens
        used_cache_cat = question_category_resp.used_cache
    else:
        question_category = ['all']
        user_interaction_id_cat = None
        n_input_tokens_cat = None
        n_output_tokens_cat = None
        used_cache_cat = None
    if len(question_category) == 1 and question_category[0] == 'irrelevant':
        model_answer = 'Please ask question related to using RenCorner commands'
        user_interaction_id_answer = None
        n_input_tokens_answer = None
        n_output_tokens_answer = None
        used_cache_answer = None
    else:
        model_answer_resp = \
            assistance.answer_user_question(user_question=user_question, question_category=question_category)
        model_answer = model_answer_resp.model_content
        user_interaction_id_answer = model_answer_resp.user_interaction_id
        n_input_tokens_answer = model_answer_resp.n_input_tokens
        n_output_tokens_answer = model_answer_resp.n_output_tokens
        used_cache_answer = model_answer_resp.used_cache
    answer_data = Answer(
        question_category=question_category,
        user_interaction_id_cat=user_interaction_id_cat,
        n_input_tokens_cat=n_input_tokens_cat,
        n_output_tokens_cat=n_output_tokens_cat,
        used_cache_cat=used_cache_cat,
        model_answer=model_answer,
        user_interaction_id_answer=user_interaction_id_answer,
        n_input_tokens_answer=n_input_tokens_answer,
        n_output_tokens_answer=n_output_tokens_answer,
        used_cache_answer=used_cache_answer
    )
    return answer_data
