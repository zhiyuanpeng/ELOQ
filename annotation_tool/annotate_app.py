import streamlit as st
import pandas as pd
import os
from os.path import join, exists


experiment_folder = os.getcwd() + '/experiment'


@st.cache_data
def load_csv_data(data_dir):
    doc_data = pd.read_csv(join(data_dir, "docs_out.csv")) 
    qrc_data = pd.read_csv(join(data_dir, "qrc_out.csv")) 
    return doc_data, qrc_data

def init():
    st.set_page_config(layout="wide")
    cwd = os.getcwd() 
    st.title("Defuse Project Data Labeling")

    if "document_content" not in st.session_state:
        st.session_state.document_content = ""
    if "question_content" not in st.session_state:
        st.session_state.question_content = ""
        
    st.markdown(
        """
        <style>
        /* Make the left column sticky */
        div[data-testid="column"]:nth-child(1) {
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            padding-right: 10px;
            padding-bottom: 30px;
            border-right: 2px solid #ccc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    return cwd 

def sidebar_logic(cwd, experiment_folder):
    st.sidebar.header("Which Experiment/Topic to Work On")
    base_path = join(cwd, experiment_folder)
    # Get the list of experiment folders 
    try:
        exp_folders = [f for f in os.listdir(base_path) if os.path.isdir(join(base_path, f))]
    except FileNotFoundError:
        st.error("The base path does not exist.")
        exp_folders = []
    # Dropdown for selecting experiment folder
    if exp_folders:
        exp_name = st.sidebar.selectbox("Choose Experiment Name:", exp_folders)
        experiment_dir = join(base_path, exp_name)
    else:
        st.error("No experiments found in the base path.")
        st.stop()

    # Get the list of topic folders
    try: 
        topic_folders = [f for f in os.listdir(experiment_dir) if os.path.isdir(join(experiment_dir, f))]
    except FileNotFoundError:
        st.error("The experiment path does not exist.")
        topic_folders = []
    # Dropdown for selecting topic folder
    if topic_folders: 
        topic = st.sidebar.selectbox("Choose Topic:", topic_folders)
        data_dir = join(experiment_dir, topic)
    else:
        st.error("No topics found in the experiment path")
        st.stop()
        
    return exp_name, data_dir

def check_username_csv_path(cwd, exp_name):
    if 'annotator_name' not in st.session_state:
        annotator_name = st.text_input("Enter your Username:", key="annotator_name_input")
        if annotator_name:
            st.session_state.annotator_name = annotator_name
    else:
        annotator_name = st.session_state.annotator_name

    if not annotator_name:
        st.warning("Please enter your Username to proceed.")
        st.stop()
    
    csv_filename = f"{annotator_name}_{exp_name}_labels.csv"
    csv_path = join(cwd, csv_filename)
    check_and_create_annotations_csv(csv_path)
    
    return csv_path

def check_and_create_annotations_csv(csv_path):
    # Check if the CSV file exists
    if not exists(csv_path):
        st.warning(f"CSV does not yet exist: {csv_path}")

        # Check if the user has made a choice already
        if 'create_csv_choice' not in st.session_state:
            st.write("### Do you want to create this new annotation CSV file? Make sure you have entered the correct Experiment Name and Username")
            choice = st.selectbox(
                "Please select an option:",
                ("Select an option", "Yes, create the file", "No, do not create")
            )
            if choice == "Yes, create the file":
                st.session_state.create_csv_choice = 'Yes'
                st.rerun()
            elif choice == "No, do not create":
                st.session_state.create_csv_choice = 'No'
                st.rerun()
            else:
                st.stop()
        else:
            if st.session_state.create_csv_choice == 'Yes':
                # Create the DataFrame with the specified columns
                columns = [
                    'doc_id', 'q_id', 'supposed_to_be_confusing', 'llm_confuse_label', 
                    'human_confuse_label', 'human_defuse_label'
                ]
                annotations_df = pd.DataFrame(columns=columns)
                # Save the DataFrame as a CSV file
                annotations_df.to_csv(csv_path, index=False)
                st.success(f"{csv_path} created successfully.")
                return csv_path
            else:
                st.warning("No CSV file created. Cannot proceed without a CSV file. Refresh to restart")
                st.stop()
    else:
        st.success(f"Happy annotating!")

def select_doc_id_with_checkmarks(doc_data, qrc_data, annotations_df):
    doc_ids = doc_data["doc_id"].unique()
    
    doc_id_labels = []
    doc_id_mapping = {}
    for doc_id in doc_ids:
        is_fully_annotated = check_if_document_fully_annotated(annotations_df, doc_id, qrc_data, return_bool=True)
        if is_fully_annotated:
            label = f"✅ {doc_id}"
        else:
            label = f"{doc_id}"
        doc_id_labels.append(label)
        doc_id_mapping[label] = doc_id

    selected_label = st.sidebar.selectbox("Choose doc_id:", doc_id_labels)
    doc_id = doc_id_mapping[selected_label]
    return doc_id

def show_instructions():
    st.write("### Instructions:")
    st.write('''Read the document, take your time and understand what it's talking about.''')
    st.write('''Questions are generated by gpt4o-mini, they either can be answered based on the document, or it can't. ''')
    st.write('''Fill out the form for each question, then click "Submit", after finishing all questions for this document, move on to the next document by selecting "Choose doc_id" on the left sidebar.''')
    st.write('''Sometimes the submit button needs to be clicked twice to record your response. If you see "Annotation submitted" or "Overwritten previous annotation", your response has been recorded.''')

def show_doc_contents(doc_data, doc_id):
    st.session_state.document_content = doc_data[doc_data["doc_id"] == doc_id]["document"].values[0]
    if st.session_state.document_content:
        st.write(f"### Document: {doc_id}")
        st.write(st.session_state.document_content)

def append_row_to_csv(csv_path, row_data, qrc_data):
    annotations_df = pd.read_csv(csv_path)
    existing_entry_index = annotations_df[
        (annotations_df['doc_id'] == row_data['doc_id']) &
        (annotations_df['q_id'] == row_data['q_id']) & 
        (annotations_df['supposed_to_be_confusing'] == row_data['supposed_to_be_confusing'])
    ].index
    
    if not existing_entry_index.empty:
        new_row_df = pd.DataFrame([row_data], columns=annotations_df.columns, index=existing_entry_index)
        annotations_df.loc[existing_entry_index] = new_row_df
        annotations_df.to_csv(csv_path, index=False)
        st.info(f"Overwritten previous annotation.")
    else:
        new_row = pd.DataFrame([row_data], columns=annotations_df.columns)
        annotations_df = pd.concat([annotations_df, new_row], ignore_index=True)
        annotations_df.to_csv(csv_path, index=False)
        st.success(f"Annotation submitted.")
        
    check_if_document_fully_annotated(annotations_df, row_data['doc_id'], qrc_data)

def check_if_document_fully_annotated(annotations_df, doc_id, qrc_data, return_bool=False):
    selected_qrc = qrc_data[(qrc_data["doc_id"] == doc_id)]
    selected_qrc = selected_qrc.sample(frac=1, random_state=42).reset_index(drop=True)
    
    question_mapping = {}
    for index, row in selected_qrc.iterrows():
        question_mapping[(row['q_id'], row['is_confusing'])] = index + 1

    annotated_questions = set(
        zip(
            annotations_df[annotations_df['doc_id'] == doc_id]['q_id'],
            annotations_df[annotations_df['doc_id'] == doc_id]['supposed_to_be_confusing']
        )
    )

    all_questions = set(
        zip(
            selected_qrc['q_id'],
            selected_qrc['is_confusing']
        )
    )
    remaining_questions = all_questions - annotated_questions
    if not remaining_questions:
        if return_bool:
            return True
        st.success(f"All questions for Document ID {doc_id} have been annotated.")
    else:
        if return_bool:
            return False
        remaining_question_indexes = [question_mapping[question] for question in remaining_questions]
        st.info(f"Question # not yet annotated: {sorted(remaining_question_indexes)} for Document {doc_id}")


# NEED TO CLICK SUBMIT TWICE
def show_question_contents_and_annotation_form(qrc_data, doc_id, csv_path, annotations_df):
    selected_qrc = qrc_data[qrc_data["doc_id"] == doc_id]
    selected_qrc = selected_qrc.sample(frac=1, random_state=42).reset_index(drop=True)

    if not selected_qrc.empty:
        for index, row in selected_qrc.iterrows():
            q_id = row['q_id']
            supposed_to_be_confusing = row['is_confusing']
            st.write(f"**Question #{index + 1}**:")
            llm_confuse_label = row['confusion'].split("\n")[0]

            # Display the Question
            st.text_area("Question:", value=row['question'], key=f"question_{index}")

            # Define keys for session state
            human_confuse_label_key = f"human_confuse_label_{index}"
            form1_submitted_key = f"form1_submitted_{index}"

            # Initialize form1_submitted in session_state if not set
            if form1_submitted_key not in st.session_state:
                st.session_state[form1_submitted_key] = False

            if not st.session_state[form1_submitted_key]:
                # Form 1: 
                with st.form(key=f'form1_{index}'):
                    human_confuse_label_options = ["Did not select", "Yes", "No"]
                    human_confuse_label = st.radio(
                        "Can this question be answered using the document? (Please select Yes or No)",
                        human_confuse_label_options,
                        key=human_confuse_label_key,
                    )
                    submit_button = st.form_submit_button(label='Submit')
                    if submit_button:
                        if human_confuse_label == "Did not select":
                            st.info("Please select 'Yes' or 'No' before submitting.")
                        else:
                            st.session_state[form1_submitted_key] = True
                            if human_confuse_label == "No":
                                # Save data immediately
                                row_data = {
                                    'doc_id': doc_id,
                                    'q_id': q_id,
                                    'supposed_to_be_confusing': supposed_to_be_confusing,
                                    'llm_confuse_label': llm_confuse_label,
                                    'human_confuse_label': human_confuse_label,
                                    'human_defuse_label': "Did not select",
                                }
                                append_row_to_csv(csv_path, row_data, qrc_data)
                            elif human_confuse_label == "Yes":
                                # Proceed to next form
                                pass
            else:
                human_confuse_label = "Yes"
                # Since form1 was submitted and human_confuse_label is "Yes", proceed to next form
                st.text_area("Response:", value=row['response'], key=f"response_{index}")
                # Form 2: Additional annotations
                with st.form(key=f'form2_{index}', clear_on_submit=True):
                    st.write("Since you think the question cannot be answered:")
                    human_defuse_label_options = ["Did not select", "Yes", "No"]
                    human_defuse_label = st.radio(
                        "Did the LLM's response defuse the confusion?",
                        human_defuse_label_options,
                        key=f"human_defuse_label_{index}",
                    )
                    submit_button = st.form_submit_button(label='Submit')
                    if submit_button:
                        if human_defuse_label == "Did not select":
                            st.info("Please select 'Yes' or 'No' before submitting.")
                        else:
                            # When the submit button is clicked, append the data to the CSV
                            row_data = {
                                'doc_id': doc_id,
                                'q_id': q_id,
                                'supposed_to_be_confusing': supposed_to_be_confusing,
                                'llm_confuse_label': llm_confuse_label,
                                'human_confuse_label': human_confuse_label,
                                'human_defuse_label': human_defuse_label,
                            }
                            append_row_to_csv(csv_path, row_data, qrc_data)
            st.write("---")  # Add a separator between questions
    else:
        st.write("No data found for the selected document and confusion status.")

if __name__ == "__main__":

    cwd = init() 

    # Sidebar logic to select experiment and topic
    exp_name, data_dir = sidebar_logic(cwd, experiment_folder)

    # Now we can get the CSV path since we have exp_name
    csv_path = check_username_csv_path(cwd, exp_name)

    # Load data
    doc_data, qrc_data = load_csv_data(data_dir)

    # Load annotations DataFrame
    if exists(csv_path):
        annotations_df = pd.read_csv(csv_path)
    else:
        columns = [
            'doc_id', 'q_id', 'supposed_to_be_confusing', 'llm_confuse_label',
            'human_confuse_label', 'human_defuse_label'
        ]
        annotations_df = pd.DataFrame(columns=columns)

    # Select doc_id with checkmarks
    doc_id = select_doc_id_with_checkmarks(doc_data, qrc_data, annotations_df)

    left, right = st.columns([2 , 1.5])  # these numbers represent proportions

    with left:
        show_instructions()
        show_doc_contents(doc_data, doc_id)
        
    with right:
        show_question_contents_and_annotation_form(qrc_data, doc_id, csv_path, annotations_df)
