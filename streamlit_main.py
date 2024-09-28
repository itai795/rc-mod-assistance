import csv
from dataclasses import asdict
import pathlib
import streamlit as st
from answer import Answer, answer_a_question

ROOT_DIR = pathlib.Path(__file__).parent


def disable_chat_input():
    st.session_state['new_user_question_disabled'] = True


def main_gpt():
    st.title('EricGPT')
    with st.expander('Commands'):
        with open(ROOT_DIR / 'documents' / 'rc_server_commands_all.txt', 'r') as f:
            all_commands_txt = f.read()
        st.text(all_commands_txt)
    previous_user_question = st.session_state.get('user_question', None)
    user_question = st.chat_input('Ask EricGPT a question',
                                  disabled=st.session_state.get('new_user_question_disabled', False),
                                  on_submit=disable_chat_input)
    if previous_user_question:
        user_question = previous_user_question
    if not user_question or user_question == '' or user_question.isspace():
        with st.chat_message('user'):
            st.markdown('')
        return

    with st.chat_message('user'):
        st.markdown(user_question)
        st.session_state['user_question'] = user_question

    if 'answer_data' in st.session_state:
        answer_data = Answer(**st.session_state['answer_data'])
    else:
        answer_data = answer_a_question(user_question=user_question)
        st.session_state['answer_data'] = asdict(answer_data)

    with st.chat_message('assistant'):
        st.markdown(answer_data.model_answer)

    with st.popover('Not the right answer?'):
        st.write('Please provide some feedback so I can improve Eric.')
        feedback_name = st.text_input('Your name (optional)')
        feedback_complaint = st.text_input('What went wrong, or what did you expect the answer to be?')
        feedback_submit = st.button('Submit')

    if feedback_submit:
        feedback_data = {
            'user_interaction_id_cat': answer_data.user_interaction_id_cat,
            'user_interaction_id_answer': answer_data.user_interaction_id_answer,
            'user_name': feedback_name,
            'user_feedback': feedback_complaint
        }
        with open(ROOT_DIR / 'cache' / 'feedback.csv', 'a', newline='\n') as f:
            writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
            writer.writerow(feedback_data)
        st.success('Feedback received, ty sir.')

    if st.button('Ask a new question'):
        del st.session_state['user_question'], st.session_state['answer_data']
        del st.session_state['new_user_question_disabled']
        st.rerun()

    with st.expander('Debugging data'):
        if answer_data.user_interaction_id_cat is not None:
            st.write(f'User interaction ID for category: {answer_data.user_interaction_id_cat}')
            st.write(f'Used cache for category question: {answer_data.used_cache_cat}')
            st.write(f'# input tokens for category question: {answer_data.n_input_tokens_cat}')
            st.write(f'# output tokens for category question: {answer_data.n_output_tokens_cat}')
        st.write(f'User interaction ID for answer: {answer_data.user_interaction_id_answer}')
        st.write(f'Used cache for answer question: {answer_data.used_cache_answer}')
        st.write(f'\# input tokens for answer question: {answer_data.n_input_tokens_answer}')
        st.write(f'\# output tokens for answer question: {answer_data.n_output_tokens_answer}')

def main():
    tab_gpt, tab_console_cmds, tab_about = st.tabs(['EricGPT', 'Console commands', 'About'])
    with tab_console_cmds:
        st.title('Console Commands')
        with st.expander('Commands'):
            with open(ROOT_DIR / 'documents' / 'rc_server_console_commands.txt', 'r') as f:
                console_commands_txt = f.read()
            st.text(console_commands_txt)
    with tab_about:
        st.title('About')
        st.markdown("""EricGPT is an AI assistant designed to help answer questions about commands on the RenCorner
server. It uses Llama 3.1, an open-source language model by Meta, hosted for free on
[openrouter.ai](https://openrouter.ai/).

A list of the server commands that EricGPT is aware of can be found in the "EricGPT" tab. Additional server commands
are available in the "Console commands" tab, however these are not referenced by EricGPT due to their disorganized order.

Please note that your questions are saved for two reasons: first, to retrieve previously asked questions from memory
instead of regenerating them; and second, to help improve EricGPT in the future.

EricGPT is not intended as a chat service, so please ask one question at a time. To ask another question, click the "Ask
a new question" button to reload the page. Since this tool is for internal use by RC moderators and has not undergone
extensive testing, please be patient—avoid clicking buttons too frequently, as EricGPT may be processing your request.
Also, avoid submitting feedback multiple times, as it will be logged each time and make a mess :).""")
    with tab_gpt:
        main_gpt()
