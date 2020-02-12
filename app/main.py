from ariadne import ObjectType, QueryType, MutationType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from asgi_lifespan import Lifespan, LifespanMiddleware
from graphqlclient import GraphQLClient
import json
# HTTP request library for access token call
import requests
# .env
from dotenv import load_dotenv
import os

# Transformers
from transformers import BertModel, BertTokenizer

# Load environment variables
load_dotenv()


def getAuthToken():
    authProvider = os.getenv('AUTH_PROVIDER')
    authDomain = os.getenv('AUTH_DOMAIN')
    authClientId = os.getenv('AUTH_CLIENT_ID')
    authSecret = os.getenv('AUTH_SECRET')
    authIdentifier = os.getenv('AUTH_IDENTIFIER')

    # Short-circuit for 'no-auth' scenario.
    if(authProvider == ''):
        print('Auth provider not set. Aborting token request...')
        return None

    url = ''
    if authProvider == 'keycloak':
        url = f'{authDomain}/auth/realms/{authIdentifier}/protocol/openid-connect/token'
    else:
        url = f'https://{authDomain}/oauth/token'

    payload = {
        'grant_type': 'client_credentials',
        'client_id': authClientId,
        'client_secret': authSecret,
        'audience': authIdentifier
    }

    headers = {'content-type': 'application/x-www-form-urlencoded'}

    r = requests.post(url, data=payload, headers=headers)
    response_data = r.json()
    print("Finished auth token request...")
    return response_data['access_token']


def getClient():

    graphqlClient = None

    # Build as closure to keep scope clean.

    def buildClient(client=graphqlClient):
        # Cached in regular use cases.
        if (client is None):
            print('Building graphql client...')
            token = getAuthToken()
            if (token is None):
                # Short-circuit for 'no-auth' scenario.
                print('Failed to get access token. Abandoning client setup...')
                return None
            url = os.getenv('MAANA_ENDPOINT_URL')
            client = GraphQLClient(url)
            client.inject_token('Bearer '+token)
        return client
    return buildClient()


models = [

    # BERT
    {"architecture": {"id": "BERT"}, "id": "bert-base-uncased",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on lower-cased English text."},
    {"architecture": {"id": "BERT"}, "id": "bert-large-uncased",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters. Trained on lower-cased English text."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-cased",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased English text."},
    {"architecture": {"id": "BERT"}, "id": "bert-large-cased",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters. Trained on cased English text."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-multilingual-uncased",
        "description": "(Original, not recommended) 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on lower-cased text in the top 102 languages with the largest Wikipedias"},
    {"architecture": {"id": "BERT"}, "id": "bert-base-multilingual-cased",
        "description": "(New, recommended) 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased text in the top 104 languages with the largest Wikipedias."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-chinese",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased Chinese Simplified and Traditional text."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-german-cased",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased German text by Deepset.ai"},
    {"architecture": {"id": "BERT"}, "id": "bert-large-uncased-whole-word-masking",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters. Trained on lower-cased English text using Whole-Word-Masking"},
    {"architecture": {"id": "BERT"}, "id": "bert-large-cased-whole-word-masking",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters. Trained on cased English text using Whole-Word-Masking."},
    {"architecture": {"id": "BERT"}, "id": "bert-large-uncased-whole-word-masking-finetuned-squad",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters. The bert-large-uncased-whole-word-masking model fine-tuned on SQuAD"},
    {"architecture": {"id": "BERT"}, "id": "bert-large-cased-whole-word-masking-finetuned-squad",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters. The bert-large-cased-whole-word-masking model fine-tuned on SQuAD"},
    {"architecture": {"id": "BERT"}, "id": "bert-base-cased-finetuned-mrpc",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. The bert-base-cased model fine-tuned on MRPC."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-german-dbmdz-cased",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased German text by DBMDZ"},
    {"architecture": {"id": "BERT"}, "id": "bert-base-german-dbmdz-uncased",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on uncased German text by DBMDZ"},
    {"architecture": {"id": "BERT"}, "id": "bert-base-japanese",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on Japanese text. Text is tokenized with MeCab and WordPiece."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-japanese-whole-word-masking",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters.  Trained on Japanese text using Whole-Word-Masking. Text is tokenized with MeCab and WordPiece."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-japanese-char",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on Japanese text. Text is tokenized into characters."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-japanese-char-whole-word-masking",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on Japanese text using Whole-Word-Masking. Text is tokenized into characters."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-finnish-cased-v1",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased Finnish text."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-finnish-uncased-v1",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on uncased Finnish text."},
    {"architecture": {"id": "BERT"}, "id": "bert-base-dutch-cased",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased Dutch text."},

    # GPT
    {"architecture": {"id": "GPT"}, "id": "openai-gpt",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. OpenAI GPT English model."},

    # GPT-2
    {"architecture": {"id": "GPT-2"}, "id": "gpt2",
        "description": "12-layer, 768-hidden, 12-heads, 117M parameters. OpenAI GPT-2 English model."},
    {"architecture": {"id": "GPT-2"}, "id": "gpt2-medium",
        "description": "24-layer, 1024-hidden, 16-heads, 345M parameters. OpenAI’s Medium-sized GPT-2 English model."},
    {"architecture": {"id": "GPT-2"}, "id": "gpt2-large",
        "description": "36-layer, 1280-hidden, 20-heads, 774M parameters. OpenAI’s Large-sized GPT-2 English model."},
    {"architecture": {"id": "GPT-2"}, "id": "gpt2-xl",
        "description": "48-layer, 1600-hidden, 25-heads, 1558M parameters. OpenAI’s XL-sized GPT-2 English model."},

    # Transformer-XL
    {"architecture": {"id": "Transformer-XL"}, "id": "transfo-xl-wt103",
        "description": "18-layer, 1024-hidden, 16-heads, 257M parameters. English model trained on wikitext-103."},

    # XL Net
    {"architecture": {"id": "XLNet"}, "id": "xlnet-base-cased",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. XLNet English model."},
    {"architecture": {"id": "XLNet"}, "id": "xlnet-large-cased",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters. XLNet Large English model"},

    # XLM
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-en-2048",
        "description": "12-layer, 2048-hidden, 16-heads. XLM English model."},
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-ende-1024",
        "description": "6-layer, 1024-hidden, 8-heads. XLM English-German model trained on the concatenation of English and German wikipedia."},
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-enfr-1024",
        "description": "6-layer, 1024-hidden, 8-heads. XLM English-French model trained on the concatenation of English and French wikipedia."},
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-enro-1024",
        "description": "6-layer, 1024-hidden, 8-heads. XLM English-Romanian Multi-language model."},
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-xnli15-1024",
        "description": "12-layer, 1024-hidden, 8-heads. XLM Model pre-trained with MLM on the 15 XNLI languages.."},
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-tlm-xnli15-1024",
        "description": "12-layer, 1024-hidden, 8-heads. XLM Model pre-trained with MLM + TLM on the 15 XNLI languages.."},
    {"architecture": {"id": "XLM"}, "id": "xlm-clm-enfr-1024",
        "description": "6-layer, 1024-hidden, 8-heads. XLM English-French model trained with CLM (Causal Language Modeling) on the concatenation of English and French wikipedia."},
    {"architecture": {"id": "XLM"}, "id": "xlm-clm-ende-1024",
        "description": "6-layer, 1024-hidden, 8-heads. XLM English-German model trained with CLM (Causal Language Modeling) on the concatenation of English and German wikipedia"},
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-17-1280",
        "description": "16-layer, 1280-hidden, 16-heads. XLM model trained with MLM (Masked Language Modeling) on 17 languages."},
    {"architecture": {"id": "XLM"}, "id": "xlm-mlm-100-1280",
        "description": "16-layer, 1280-hidden, 16-heads. xXLM model trained with MLM (Masked Language Modeling) on 100 languages."},

    # RoBERTa
    {"architecture": {"id": "RoBERTa"}, "id": "roberta-base",
        "description": "12-layer, 768-hidden, 12-heads, 125M parameters. RoBERTa using the BERT-base architecture."},
    {"architecture": {"id": "RoBERTa"}, "id": "roberta-large",
        "description": "24-layer, 1024-hidden, 16-heads, 355M parameters. RoBERTa using the BERT-large architecture."},
    {"architecture": {"id": "RoBERTa"}, "id": "roberta-large-mnli",
        "description": "24-layer, 1024-hidden, 16-heads, 355M parameters roberta-large fine-tuned on MNLI."},
    {"architecture": {"id": "RoBERTa"}, "id": "distilroberta-base",
        "description": "6-layer, 768-hidden, 12-heads, 82M parameters. The DistilRoBERTa model distilled from the RoBERTa model roberta-base checkpoint."},
    {"architecture": {"id": "RoBERTa"}, "id": "roberta-base-openai-detector",
        "description": "12-layer, 768-hidden, 12-heads, 125M parameters. roberta-base fine-tuned by OpenAI on the outputs of the 1.5B-parameter GPT-2 model."},
    {"architecture": {"id": "RoBERTa"}, "id": "roberta-large-openai-detector",
        "description": "24-layer, 1024-hidden, 16-heads, 355M parameters. roberta-large fine-tuned by OpenAI on the outputs of the 1.5B-parameter GPT-2 model."},

    # DistilBERT
    {"architecture": {"id": "DistilBERT"}, "id": "distilbert-base-uncased",
        "description": "6-layer, 768-hidden, 12-heads, 66M parameters. The DistilBERT model distilled from the BERT model bert-base-uncased checkpoint."},
    {"architecture": {"id": "DistilBERT"}, "id": "distilbert-base-uncased-distilled-squad",
        "description": "6-layer, 768-hidden, 12-heads, 66M parameters. The DistilBERT model distilled from the BERT model bert-base-uncased checkpoint, with an additional linear layer."},
    {"architecture": {"id": "DistilBERT"}, "id": "distilgpt2",
        "description": "6-layer, 768-hidden, 12-heads, 82M parameters. The DistilGPT2 model distilled from the GPT2 model gpt2 checkpoint."},
    {"architecture": {"id": "DistilBERT"}, "id": "distilbert-base-german-cased",
        "description": "6-layer, 768-hidden, 12-heads, 66M parameters. The German DistilBERT model distilled from the German DBMDZ BERT model bert-base-german-dbmdz-cased checkpoint."},
    {"architecture": {"id": "DistilBERT"}, "id": "distilbert-base-multilingual-cased",
        "description": "6-layer, 768-hidden, 12-heads, 134M parameters. The multilingual DistilBERT model distilled from the Multilingual BERT model bert-base-multilingual-cased checkpoint.."},

    # CTRL
    {"architecture": {"id": "CTRL"}, "id": "ctrl",
        "description": "48-layer, 1280-hidden, 16-heads, 1.6B parameters. Salesforce’s Large-sized CTRL English model."},

    # CamemBERT
    {"architecture": {"id": "CamemBERT"}, "id": "camembert-base",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters. CamemBERT using the BERT-base architecture."},

    # ALBERT
    {"architecture": {"id": "ALBERT"}, "id": "albert-base-v1",
        "description": "12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters. ALBERT base model."},
    {"architecture": {"id": "ALBERT"}, "id": "albert-large-v1",
        "description": "24 repeating layers, 128 embedding, 1024-hidden, 16-heads, 17M parameters. ALBERT large model."},
    {"architecture": {"id": "ALBERT"}, "id": "albert-xlarge-v1",
        "description": "24 repeating layers, 128 embedding, 2048-hidden, 16-heads, 58M parameters. ALBERT xlarge model."},
    {"architecture": {"id": "ALBERT"}, "id": "albert-xxlarge-v1",
        "description": "12 repeating layer, 128 embedding, 4096-hidden, 64-heads, 223M parameters. ALBERT xxlarge model."},
    {"architecture": {"id": "ALBERT"}, "id": "albert-base-v2",
        "description": "12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters. ALBERT base model with no dropout, additional training data and longer training."},
    {"architecture": {"id": "ALBERT"}, "id": "albert-large-v2",
        "description": "24 repeating layers, 128 embedding, 1024-hidden, 16-heads, 17M parameters. ALBERT large model with no dropout, additional training data and longer training."},
    {"architecture": {"id": "ALBERT"}, "id": "albert-xlarge-v2",
        "description": "24 repeating layers, 128 embedding, 2048-hidden, 16-heads, 58M parameters. ALBERT xlarge model with no dropout, additional training data and longer training."},
    {"architecture": {"id": "ALBERT"}, "id": "albert-xxlarge-v2",
        "description": "12 repeating layer, 128 embedding, 4096-hidden, 64-heads, 223M parameters. ALBERT xxlarge model with no dropout, additional training data and longer training."},

    # T5
    {"architecture": {"id": "T5"}, "id": "t5-small",
        "description": "~60M parameters with 6-layers, 512-hidden-state, 2048 feed-forward hidden-state, 8-heads. Trained on English text: the Colossal Clean Crawled Corpus (C4)."},
    {"architecture": {"id": "T5"}, "id": "t5-base",
        "description": "~220M parameters with 12-layers, 768-hidden-state, 3072 feed-forward hidden-state, 12-heads. Trained on English text: the Colossal Clean Crawled Corpus (C4)."},
    {"architecture": {"id": "T5"}, "id": "t5-large",
        "description": "~770M parameters with 24-layers, 1024-hidden-state, 4096 feed-forward hidden-state, 16-heads. Trained on English text: the Colossal Clean Crawled Corpus (C4)."},
    {"architecture": {"id": "T5"}, "id": "t5-3B",
        "description": "~2.8B parameters with 24-layers, 1024-hidden-state, 16384 feed-forward hidden-state, 32-heads. Trained on English text: the Colossal Clean Crawled Corpus (C4)."},
    {"architecture": {"id": "T5"}, "id": "t5-11B",
        "description": "~11B parameters with 24-layers, 1024-hidden-state, 65536 feed-forward hidden-state, 128-heads. Trained on English text: the Colossal Clean Crawled Corpus (C4)."},

    # XLM-RoBERTa
    {"architecture": {"id": "XLM-RoBERTa"}, "id": "xlm-roberta-base",
        "description": "~125M parameters with 12-layers, 768-hidden-state, 3072 feed-forward hidden-state, 8-heads. Trained on on 2.5 TB of newly created clean CommonCrawl data in 100 languages."},
    {"architecture": {"id": "XLM-RoBERTa"}, "id": "xlm-roberta-large",
        "description": "~355M parameters with 24-layers, 1027-hidden-state, 4096 feed-forward hidden-state, 16-heads. Trained on 2.5 TB of newly created clean CommonCrawl data in 100 languages."},

    # FlauBERT
    {"architecture": {"id": "FlauBERT"}, "id": "flaubert-small-cased",
        "description": "6-layer, 512-hidden, 8-heads, 54M parameters. FlauBERT small architecture."},
    {"architecture": {"id": "FlauBERT"}, "id": "flaubert-base-uncased",
        "description": "12-layer, 768-hidden, 12-heads, 137M parameters. FlauBERT base architecture with uncased vocabulary."},
    {"architecture": {"id": "FlauBERT"}, "id": "flaubert-base-cased",
        "description": "12-layer, 768-hidden, 12-heads, 138M parameters. FlauBERT base architecture with cased vocabulary."},
    {"architecture": {"id": "FlauBERT"}, "id": "flaubert-large-cased",
        "description": "24-layer, 1024-hidden, 16-heads, 373M parameters. FlauBERT large architecture."},
]

# Define types using Schema Definition Language (https://graphql.org/learn/schema/)
# Wrapping string in gql function provides validation and better error traceback
type_defs = gql("""

    type Architecture {
        id: ID!
    }

    type Model {
        id: ID!
        architecture: Architecture!
        description: String
    }

    type Query {
        models: [Model!]!
        test(text: String!): [String!]!
    }

""")

# Map resolver functions to Query fields using QueryType
query = QueryType()

# Resolvers are simple python functions
@query.field("models")
def resolve_models(_, info):
    return models


# Cache the model instance
bert_model = None
bert_tokenizer = None

# Preprocessing text for BERT
@query.field("test")
def resolve_test(*_, text):
    # 1. Load (and cache) the model and tokenizer
    global bert_model
    global bert_tokenizer
    if (bert_model == None):
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 2. Tokenize input text
    tokens = bert_tokenizer.tokenize(text)

    # 3. Add special tokens
    special_tokens = ['[CLS]'] + tokens + ['[SEP]']

    # 4. Pad
    maxlen = 12
    padded_tokens = special_tokens + \
        ['[PAD]' for _ in range(maxlen - len(special_tokens))]

    # 5. Attention mask
    attn_mask = [1 if tok != '[PAD]' else 0 for tok in padded_tokens]

    # 6. Segment IDs (for sequence pairs)
    token_ids = bert_tokenizer.convert_tokens_to_ids(padded_tokens)

    # 7. Convert sequence to integers
    encoding = bert_tokenizer.encode(
        "Here is some text to encode", add_special_tokens=True)
    print(encoding)

    return padded_tokens

# # Map resolver functions to custom type fields using ObjectType
# person = ObjectType("Person")


# @person.field("fullName")
# def resolve_person_fullname(person, *_):
#     return "%s %s" % (person["firstName"], person["lastName"])


# Create executable GraphQL schema
schema = make_executable_schema(type_defs, [query])  # , person])

# --- ASGI app

# Create an ASGI app using the schema, running in debug mode
# Set context with authenticated graphql client.
app = GraphQL(
    schema, debug=True, context_value={'client': getClient()})

# 'Lifespan' is a standalone ASGI app.
# It implements the lifespan protocol,
# and allows registering lifespan event handlers.
lifespan = Lifespan()


@lifespan.on_event("startup")
async def startup():
    print("Starting up...")
    print("... done!")


@lifespan.on_event("shutdown")
async def shutdown():
    print("Shutting down...")
    print("... done!")

# 'LifespanMiddleware' returns an ASGI app.
# It forwards lifespan requests to 'lifespan',
# and anything else goes to 'app'.
app = LifespanMiddleware(app, lifespan=lifespan)
