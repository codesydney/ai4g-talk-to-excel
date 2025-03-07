import asyncio
import chainlit as cl
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
import logging
import sys
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

from services.llm_factory import LLMFactory

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logging.getLogger("httpx").setLevel(logging.ERROR)

def load_language_model(llm_client: str):
    """
    Load the language model using the LLMFactory and the specified client
    - this makes it easy to swap between different language models.
    """
    Settings.llm = LLMFactory(llm_client).client

load_language_model("bedrock")

@cl.on_chat_start
async def on_chat_start():
    """
    This function is called when the user starts a new chat session.
    """
    await cl.Message(
        content=
        """
        **Welcome to Code.Sydney Excel Data Explorer!** 
        Please upload an Excel file (.xlsx or .xls) to begin analysing it.
        After uploading, you can ask questions about the data such as:
        - What is the average of a specific column?
        - How many rows are in the dataset?
        - What are the correlations between columns?
        - Plot distributions of values
        """
    ).send()
    
    files = await cl.AskFileMessage(
        content="Please upload an Excel file to analyse.",
        accept=["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"],
        max_size_mb=10,
        timeout=180,
    ).send()
    
    if not files:
        await cl.Message(
            content="No file was uploaded. Please refresh and try again."
        ).send()
        return
        
    try:
        loading_msg = await cl.Message(content="📊 Loading and processing the file...").send()
        
        spinner_states = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for _ in range(10):  # Loop for a short period to simulate the effect
            for state in spinner_states:
                loading_msg.content = f"{state} Processing file..."
                await loading_msg.update()
                await asyncio.sleep(0.1)  # Small delay to create a spinning effect
        
        file = files[0]
        print (f"Uploaded file: {file.path}")
        df = pd.read_excel(file.path)
        print(f"Number of records: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        loading_msg.content = f"✅ Uploaded: {file.name}."
        await loading_msg.update()
        
        cl.user_session.set("df", df) # Save the dataframe in the user session, for use later
        
        message = cl.Message(content=f"Successfully loaded *{file.name}* with {len(df)} rows and {len(df.columns)} columns. Here's a preview:")
        await message.send()
        
        element = cl.Dataframe(data=df.head(10), name=f"{file.name} Preview", size="large", display="inline")
        await cl.Message(content="", elements=[element]).send()
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Values': df.count().values,
            'Null Values': df.isna().sum().values,
        })
        
        col_element = cl.Dataframe(data=col_info, name="Column Information", size="medium")
        await cl.Message(content="**Column Information:**", elements=[col_element]).send()
        
        await cl.Message(
            content=
            """
            You can now ask questions about this data. For example:
            - What is the summary statistics of the dataset?
            - Show me the distribution of [column name]
            - Is there a correlation between [column1] and [column2]?
            - Create a plot showing [specific analysis]
            """
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ Error processing the uploaded file: {str(e)}"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    This function is called whenever a new message is received from the user.
    """
    import matplotlib.pyplot as plt

    print(f"Received message: {message.content}")
    df = cl.user_session.get("df")
    query_engine = PandasQueryEngine(
        df=df,
        verbose=True,
        synthesize_response=True,
        description="This dataframe contains some dataset"
    )
    
    if not query_engine:
        await cl.Message(content="The query engine has not been properly initialised.").send()
        return
    
    thinking_msg = cl.Message(content="Analyzing the data...")
    await thinking_msg.send()
    
    try:
        print(f"Before predict")
        # Use the language model to check if the message is asking for a plot or visualization
        prediction = Settings.llm.predict(
            prompt=PromptTemplate(f"Does the following message explicitly ask for a graph, plot or visualisation? \
                {message.content}. Answer Yes or No only."),
            max_tokens=10,
        )
        print(f"Plot or graph needed: {prediction}")
        print(f"Querying: {message.content}")
        
        plt.clf() # Clear the current plot before generating a new one
        
        # This will call eval() on the pandas instruction generated by the LLM
        response = query_engine.query(message.content)
        
        if "yes" in prediction.lower():
            print("Generating plot...")
            plt.savefig('new_plot.png')
            
            elements = [
                cl.Image(name="Generated Plot", path="new_plot.png")
            ]
            
            thinking_msg.content = f"{response}"
            thinking_msg.elements = elements
            await thinking_msg.update()
            
            await cl.Message(content="Here is the generated plot:", elements=elements).send()
        else:
            print("No plot needed")
            
            thinking_msg.content = f"{response}"
            await thinking_msg.update()
        
    except Exception as e:
        thinking_msg.content = f"Error processing your question: {str(e)}"
        await thinking_msg.update()