from flask import Flask, request, jsonify,render_template
from langchain import OpenAI
import os
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index import set_global_service_context

# Initialize Flask app
app = Flask(__name__)
openai_api_key=os.getenv("API")

# Set up OpenAI
llm = OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1, openai_api_key=os.getenv("API"))

# Set up the service context
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context=service_context)

# Load the data from connectors.csv and create the index
connector_csv = SimpleDirectoryReader(input_files=["connectors.csv"]).load_data()
connector_index = VectorStoreIndex.from_documents(connector_csv)
connector_engine = connector_index.as_query_engine(similarity_top_k=3)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
async def get_response():
    try:
        # Get the user input from the request's JSON data
        user_input = request.json.get('user_input')

        # Perform the query on the connector_engine with await
        response = await connector_engine.aquery(f'get connector, action and description for : {user_input}, the output should strictly follow json format with keys : connector, action , description')
        result=[]
        result.append(response.response)
        # Return the result as a JSON response
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port)
