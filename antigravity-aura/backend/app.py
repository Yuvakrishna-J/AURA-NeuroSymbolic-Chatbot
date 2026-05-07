from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from train_chatbot import MentalHealthChatbot
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re
import unicodedata
from difflib import SequenceMatcher
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize ENHANCED ML-based distress detector (v2 with feature engineering)
distress_detector = None
try:
    from distress_detector_v2 import EnhancedMentalHealthDetector
    distress_detector = EnhancedMentalHealthDetector(model_type='random_forest')
    
    # Try v2 model first (enhanced with 78.85% accuracy)
    model_path_v2 = os.path.join(os.path.dirname(__file__), 'distress_detector_v2_random_forest.pkl')
    model_path_v1 = os.path.join(os.path.dirname(__file__), 'distress_detector.pkl')
    
    if os.path.exists(model_path_v2):
        distress_detector.load_model(model_path_v2)
        print("✅ ENHANCED ML distress detector v2 loaded (RandomForest + Features)")
        # Display test metrics if available
        test_metrics = distress_detector.get_test_metrics()
        if test_metrics and isinstance(test_metrics, dict):
            print(f"   📊 Gold Standard Test Performance:")
            print(f"      Accuracy:  {test_metrics['accuracy']*100:.2f}%")
            print(f"      Precision: {test_metrics['precision']*100:.2f}%")
            print(f"      Recall:    {test_metrics['recall']*100:.2f}%")
            print(f"      F1-Score:  {test_metrics['f1_score']*100:.2f}%")
            print(f"      Model:     {test_metrics['model_type'].upper()}")
            print(f"      Features:  {test_metrics['feature_count']} (TF-IDF + LIWC + Social + Sentiment)")
    else:
        print("⚠️ Enhanced model not found, training now (this may take 2-3 minutes)...")
        csv_path = os.path.join(os.path.dirname(__file__), 'train_data.csv')
        if os.path.exists(csv_path):
            distress_detector.train(csv_path, test_size=0.1, val_size=0.1)
            distress_detector.save_model(model_path_v2)
            print("✅ Enhanced ML distress detector v2 trained and saved")
        else:
            print("⚠️ train_data.csv not found, ML detection disabled")
            distress_detector = None
except Exception as e:
    print(f"⚠️ Could not load enhanced detector, trying fallback v1: {e}")
    # Fallback to v1 if v2 fails
    try:
        from distress_detector import MentalHealthDetector
        distress_detector = MentalHealthDetector()
        model_path = os.path.join(os.path.dirname(__file__), 'distress_detector.pkl')
        if os.path.exists(model_path):
            distress_detector.load_model(model_path)
            print("✅ ML distress detector v1 loaded (fallback)")
    except Exception as e2:
        print(f"⚠️ Could not load any distress detector: {e2}")
        distress_detector = None

# Load train_data.csv as primary dataset
counseling_dataset = []
import csv
import pandas as pd

try:
    # Load from train_data.csv in parent directory
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_data.csv')
    
    if os.path.exists(csv_path):
        # Read CSV using pandas for better handling
        df = pd.read_csv(csv_path)
        
        # Create context-response pairs from the distress dataset
        for _, row in df.iterrows():
            text = str(row['text']) if 'text' in row else ''
            label = int(row['label']) if 'label' in row else 0
            confidence = float(row['confidence']) if 'confidence' in row else 0.5
            sentiment = float(row['sentiment']) if 'sentiment' in row else 0.0
            
            if len(text) < 20:  # Skip very short texts
                continue
            
            # Determine category based on label and content
            category = 'crisis' if label == 1 else 'general'
            
            # Generate appropriate response based on distress level
            if label == 1:  # Distress detected
                response = ("I hear how much pain you're going through, and I want you to know that you're not alone. "
                           "What you're feeling is valid, and there are people who care about you and want to help. "
                           "Please consider reaching out to a crisis helpline immediately:\n\n"
                           "🇮🇳 India Crisis Helplines:\n"
                           "• Tele MANAS: 14416 or 1800-89-14416 (24/7, Available in 20 languages)\n"
                           "• KIRAN Mental Health: 1800-599-0019 (24/7 free)\n"
                           "• Vandrevala Foundation: 1860-2662-345\n"
                           "• iCall: 9152987821\n\n"
                           "Your life has value, and with support, things can get better. "
                           "Would you like to talk more about what you're experiencing?")
            else:
                response = ("Thank you for sharing what's on your mind. I'm here to listen and support you. "
                           "It sounds like you're dealing with something challenging. "
                           "Remember that seeking help is a sign of strength, not weakness. "
                           "Would you like to explore some coping strategies or talk more about how you're feeling?")
            
            counseling_dataset.append({
                'context': text,
                'response': response,
                'categories': [category],
                'quality_score': int(confidence * 100),
                'label': label,
                'sentiment': sentiment
            })
        
        print(f"✅ Loaded {len(counseling_dataset)} entries from train_data.csv")
        avg_quality = sum(e.get('quality_score', 0) for e in counseling_dataset) / len(counseling_dataset) if counseling_dataset else 0
        print(f"   📊 Average quality score: {avg_quality:.1f}")
        distress_count = sum(1 for e in counseling_dataset if e.get('label') == 1)
        print(f"   📊 Distress samples: {distress_count} / {len(counseling_dataset)} ({distress_count/len(counseling_dataset)*100:.1f}%)")
    else:
        print(f"⚠️ train_data.csv not found at {csv_path}, trying fallback...")
        # Fallback to processed dataset
        dataset_path = os.path.join(os.path.dirname(__file__), 'combined_dataset_processed_simple.json')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                counseling_dataset = json.load(f)
            print(f"✅ Loaded {len(counseling_dataset)} processed counseling responses (fallback)")
except Exception as e:
    print(f"⚠️ Could not load dataset: {e}")
    import traceback
    traceback.print_exc()

# Function to find best matching response from dataset (enhanced)
def find_best_counseling_response(user_message):
    """Find best matching response using intelligent scoring"""
    if not counseling_dataset:
        return None
    
    user_message_lower = user_message.lower()
    
    # Category keywords for intelligent matching
    category_keywords = {
        'crisis': ['suicide', 'suicidal', 'kill myself', 'end my life', 'die', 'death', 'cant take', "can't take"],
        'depression': ['depression', 'depressed', 'sad', 'hopeless', 'empty', 'worthless', 'useless'],
        'anxiety': ['anxiety', 'anxious', 'worried', 'panic', 'nervous', 'stress', 'overwhelmed'],
        'trauma': ['abuse', 'trauma', 'hurt', 'violated', 'ptsd', 'assault'],
        'relationships': ['relationship', 'marriage', 'partner', 'boyfriend', 'girlfriend', 'spouse', 'divorce'],
        'family': ['family', 'parent', 'mother', 'father', 'child', 'sibling', 'brother', 'sister'],
        'self-esteem': ['self esteem', 'confidence', 'worth', 'value', 'believe in myself'],
        'grief': ['grief', 'loss', 'died', 'death', 'mourning'],
        'sleep': ['sleep', 'insomnia', 'cant sleep', "can't sleep", 'tired', 'exhausted'],
        'general': ['help', 'need', 'advice', 'what should']
    }
    
    # Detect user message category
    user_categories = []
    for category, keywords in category_keywords.items():
        if any(keyword in user_message_lower for keyword in keywords):
            user_categories.append(category)
    
    if not user_categories:
        user_categories = ['general']
    
    scored_responses = []
    for entry in counseling_dataset:
        context = entry.get('context', entry.get('Context', '')).lower()
        response = entry.get('response', entry.get('Response', ''))
        entry_categories = entry.get('categories', ['general'])
        quality_score = entry.get('quality_score', 50)
        
        if not response or len(response) < 50:
            continue
        
        score = 0
        
        # Category match (high weight)
        category_matches = len(set(user_categories) & set(entry_categories))
        score += category_matches * 30
        
        # Keyword matching in context
        for category in user_categories:
            keywords = category_keywords.get(category, [])
            for keyword in keywords:
                if keyword in user_message_lower and keyword in context:
                    score += 15
        
        # Word overlap (lower weight, for general matching)
        user_words = set(user_message_lower.split())
        context_words = set(context.split())
        # Remove common words
        stop_words = {'i', 'me', 'my', 'am', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        user_words -= stop_words
        context_words -= stop_words
        overlap = len(user_words & context_words)
        score += overlap * 2
        
        # Quality boost (prefer higher quality responses)
        score += quality_score * 0.2
        
        # Priority for crisis responses
        if 'crisis' in user_categories and 'crisis' in entry_categories:
            score += 50
        
        if score > 0:
            scored_responses.append((score, response, entry_categories))
    
    if scored_responses:
        # Sort by score (highest first)
        scored_responses.sort(reverse=True, key=lambda x: x[0])
        # Return best matching response
        return scored_responses[0][1]
    
    return None

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
use_gemini = False
gemini_model = None
gemini_base_model_name = None  # Store the base model name

if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List all available models for debugging
        print("📋 Listing available models...")
        try:
            available_models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
                    print(f"   • {model.name}")
            print(f"✅ Found {len(available_models)} models supporting generateContent")
        except Exception as list_error:
            print(f"⚠️ Could not list models: {list_error}")
        
        # Using Gemma 3 / Gemini models for enhanced mental health support
        model_names = [
            'models/gemma-3-27b-it',      # Gemma 3 27B Instruction Tuned (Primary)
            'models/gemini-2.5-flash',    # Gemini 2.5 Flash (Fallback)
            'models/gemini-2.0-flash',    # Gemini 2.0 Flash (Fallback)
            'models/gemini-2.5-pro',      # Gemini 2.5 Pro (Fallback)
        ]
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                # Skip test generation - initialize model only
                # test_response = gemini_model.generate_content("Hi")
                use_gemini = True
                gemini_base_model_name = model_name  # Store for later use
                print(f"✅ Gemini AI initialized successfully with model: {model_name}")
                break
            except Exception as model_error:
                print(f"⚠️ Failed to load {model_name}: {model_error}")
                continue
        
        if not use_gemini:
            print("⚠️ All Gemini models failed to initialize")
            print("   Falling back to trained model only...")
            gemini_model = None
    except Exception as e:
        print(f"⚠️ Gemini configuration failed: {e}")
        print("   Falling back to trained model only...")
        use_gemini = False
        gemini_model = None
else:
    print("ℹ️ No Gemini API key found. Using trained model only.")

# RAG components - DISABLED by default for faster startup
# Will be loaded on first request if needed
use_rag = False
rag_collection = None
embedding_model = None
chroma_client = None

print("ℹ️ RAG system disabled for faster startup")
print("   Server will use Gemini AI for responses")

# Initialize T5 model for empathetic response generation
t5_model = None
t5_tokenizer = None
use_t5 = False

# T5 disabled for faster startup
print("ℹ️ T5 model disabled for faster startup")

# Google Translate is available globally - no initialization needed
use_translator = True
print("✅ Google Translate initialized successfully")
print("   Supports 100+ languages including Hindi, Tamil, Telugu, Marathi, etc.")
print("   Using Gemini AI for all responses")

# Initialize chatbot
chatbot = MentalHealthChatbot()

# Load the trained model
model_path = 'chatbot_model.pkl'
if os.path.exists(model_path):
    chatbot.load_model(model_path)
    print("✅ Chatbot model loaded successfully!")
else:
    print("⚠️ No trained model found. Please train the model first.")
    print("Run: python train_chatbot.py")

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'AURA Mental Health Chatbot API',
        'status': 'running',
        'endpoints': {
            'chat': '/api/chat (POST)',
            'health': '/api/health (GET)',
            'modes': '/api/modes (GET)'
        }
    })

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle chat requests with ML-based distress detection"""
    if request.method == 'OPTIONS':
        return '', 204
    
    print(f"🔵 CHAT ENDPOINT CALLED - Method: {request.method}")
    
    try:
        data = request.get_json()
        print(f"📨 Received request: {data}")
        print(f"🔧 use_gemini={use_gemini}, use_ai={data.get('useAI', True)}, mode={data.get('mode', 'friend')}")
        
        user_message = data.get('message', '')
        mode = data.get('mode', 'friend')
        use_ai = data.get('useAI', True)  # Option to use Gemini or trained model
        language = data.get('language', 'English')  # User's preferred language
        
        print(f"🌐 User language: {language}")
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # HYBRID Crisis Detection: ML Model (primary) + Keywords (backup)
        # ML model has 78.85% accuracy, keywords expanded to catch paraphrased language
        
        # Step 1: ML-based distress detection (PRIMARY)
        ml_distress_result = None
        ml_detected_crisis = False
        
        if distress_detector:
            try:
                ml_distress_result = distress_detector.predict_distress(user_message)
                print(f"🧠 ML Distress Detection: {ml_distress_result}")
                ml_detected_crisis = ml_distress_result.get('is_distress', False)
                
                if ml_detected_crisis:
                    print(f"🚨 ML MODEL DETECTED CRISIS (confidence: {ml_distress_result.get('confidence', 0):.2%})")
            except Exception as e:
                print(f"⚠️ ML distress detection error: {e}")
        
        # Step 2: Keyword-based detection (BACKUP)
        has_crisis_keywords = detect_crisis_keywords(user_message)
        if has_crisis_keywords:
            print(f"🚨 CRISIS KEYWORDS DETECTED: Immediate intervention required!")
        
        # Step 3: UNION approach - flag as crisis if EITHER detector triggers
        is_crisis = ml_detected_crisis or has_crisis_keywords
        
        if is_crisis:
            # Ensure crisis intervention is flagged
            if ml_distress_result:
                ml_distress_result['is_distress'] = True
                ml_distress_result['requires_crisis_intervention'] = True
                # Boost confidence if keywords also matched
                if has_crisis_keywords and ml_distress_result.get('confidence', 0) < 0.95:
                    ml_distress_result['confidence'] = 0.95
                    ml_distress_result['probability'] = 0.95
                    print(f"✅ Crisis confirmed by BOTH ML and keywords (high confidence)")
            else:
                # ML not available, create result from keywords
                ml_distress_result = {
                    'is_distress': True,
                    'confidence': 0.90,  # High but not 100% (keywords can have false positives)
                    'probability': 0.90,
                    'requires_crisis_intervention': True
                }
                print(f"✅ Crisis detected by keywords (ML unavailable)")
        
        # Try to find a matching response from professional counseling dataset first
        counseling_response = find_best_counseling_response(user_message)
        
        if counseling_response and mode == 'professional':
            # Use professional counseling response from dataset
            response = counseling_response
            source = 'counseling_dataset'
        elif use_gemini and use_ai:
            # Use Gemini AI with mental health context
            response = get_gemini_response(user_message, mode, language)
            source = 'gemini'
        elif use_t5:
            # Use T5 for empathetic response generation
            response = generate_t5_response(user_message, mode)
            source = 't5_model'
            # Fallback if T5 fails
            if not response:
                response = chatbot.get_response(user_message, mode=mode)
                source = 'trained_model'
        else:
            # Use trained model
            response = chatbot.get_response(user_message, mode=mode)
            source = 'trained_model'
        
        print(f"✅ Sending response ({source}): {response[:100]}...")
        
        # Build response with ML distress info
        response_data = {
            'response': response,
            'mode': mode,
            'source': source
        }
        
        # Add ML distress detection results if available
        if ml_distress_result:
            response_data['distress_detection'] = {
                'is_distress': bool(ml_distress_result['is_distress']),
                'confidence': float(ml_distress_result['confidence']),
                'distress_probability': float(ml_distress_result.get('probability', 0.0)),
                'requires_crisis_intervention': bool(ml_distress_result.get('requires_crisis_intervention', False))
            }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def detect_crisis_keywords(text):
    """Detect critical suicide/self-harm keywords for immediate intervention
    
    EXPANDED keyword list based on real-world crisis language patterns.
    Includes direct mentions, euphemisms, and paraphrased expressions.
    """
    def normalize_crisis_text(raw_text):
        """Normalize noisy user text (typos, punctuation, leetspeak) for safer matching."""
        normalized = unicodedata.normalize('NFKD', str(raw_text)).lower()

        # Common leetspeak/short-hand substitutions seen in crisis text.
        for old, new in {
            '@': 'a',
            '$': 's',
            '0': 'o',
            '1': 'i',
            '3': 'e',
            '4': 'a',
            '5': 's',
            '7': 't',
            '8': 'b'
        }.items():
            normalized = normalized.replace(old, new)

        normalized = re.sub(r"[\-_./\\|]+", " ", normalized)
        normalized = re.sub(r"[^a-z\s]", " ", normalized)

        # Compress exaggerated letters (e.g., "suuuuicide" -> "suuicide").
        normalized = re.sub(r"(.)\1{2,}", r"\1\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return {
            'spaced': normalized,
            'compact': normalized.replace(' ', ''),
            'tokens': normalized.split()
        }

    crisis_keywords = [
        # Direct suicide mentions (including common misspellings)
        'suicide', 'suicidal', 'sucide', 'suicde', 'suiside', 'sucidal',
        'kys', 'kms',
        'kill myself', 'killing myself',
        'end my life', 'ending my life', 'take my life', 'taking my life',
        
        # Death wishes
        'want to die', 'wanna die', 'wish i was dead', 'wish i were dead',
        'better off dead', 'want to be dead', 'don\'t want to live',
        'no reason to live', 'not worth living', 'life isn\'t worth',
        
        # Ending/finishing expressions
        'end it all', 'end this', 'finish myself', 'finish it',
        'can\'t go on', 'cannot go on', 'give up on life',
        
        # Self-harm
        'harm myself', 'hurt myself', 'cut myself', 'cutting myself',
        'self harm', 'self-harm', 'self injury',
        
        # Crisis methods
        'overdose', 'jump off', 'hang myself', 'hanging myself',
        'shoot myself', 'drown myself', 'pills',
        
        # Hopelessness (paraphrased crisis language)
        'no way out', 'no escape', 'trapped', 'no hope',
        'hopeless', 'helpless', 'no point in living',
        'don\'t see a point', 'no point anymore', 'pointless',
        
        # Pain/suffering expressions
        'can\'t take this', 'cannot take this', 'can\'t take it',
        'too much pain', 'unbearable', 'can\'t bear',
        'want it to stop', 'make it stop', 'end the pain',
        
        # Finality expressions  
        'saying goodbye', 'final goodbye', 'won\'t be here',
        'better without me', 'burden', 'everyone would be better',
        'disappear forever', 'cease to exist', 'stop existing'
    ]

    normalized = normalize_crisis_text(text)
    text_spaced = normalized['spaced']
    text_compact = normalized['compact']
    text_tokens = normalized['tokens']

    if not text_spaced:
        return False

    # 1) Exact keyword/phrase checks (fast path).
    for keyword in crisis_keywords:
        keyword_norm = re.sub(r"\s+", " ", keyword.lower()).strip()
        keyword_compact = keyword_norm.replace(' ', '')

        if keyword_norm in text_spaced:
            return True

        # Catches fused inputs like "killmyself".
        if keyword_compact and keyword_compact in text_compact:
            return True

    # 2) Typo-tolerant single-token matching (e.g., "suicde", "sucidel").
    high_risk_terms = {
        'suicide', 'suicidal', 'overdose', 'selfharm', 'selfinjury',
        'hang', 'hanging', 'die', 'death', 'kill', 'kys', 'kms'
    }

    compact_tokens = [tok.replace(' ', '') for tok in text_tokens]
    for token in compact_tokens:
        if len(token) < 3:
            continue
        for risk_term in high_risk_terms:
            # Strong similarity threshold to reduce false positives.
            if SequenceMatcher(None, token, risk_term).ratio() >= 0.86:
                return True

    return False

def retrieve_context(query, top_k=3):
    """Retrieve relevant context from RAG database"""
    if not use_rag:
        return ""
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Query vector database
        results = rag_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format retrieved documents
        context_docs = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            context_docs.append(f"- {doc}")
        
        return "\n".join(context_docs)
    except Exception as e:
        print(f"⚠️ RAG retrieval error: {e}")
        return ""

def generate_t5_response(user_message, mode='empathetic'):
    """Generate empathetic response using T5 model"""
    if not use_t5:
        return None
    
    try:
        import torch
        
        # Create prompt based on mode
        if mode == 'friend':
            prompt = f"respond empathetically to: {user_message}"
        elif mode == 'counselor':
            prompt = f"provide mental health support for: {user_message}"
        else:
            prompt = f"offer compassionate response to: {user_message}"
        
        # Tokenize input
        input_ids = t5_tokenizer(
            prompt,
            return_tensors='pt',
            max_length=128,
            truncation=True
        ).input_ids.to(t5_model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = t5_model.generate(
                input_ids,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=False,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        print(f"⚠️ T5 generation error: {e}")
        return None

def translate_to_language(text, target_language):
    """
    Translate English text to target language using Google Translate (deep-translator)
    
    Args:
        text: English text to translate
        target_language: Target language name (e.g., 'Hindi', 'Tamil', 'Telugu')
    
    Returns:
        Translated text or original if translation fails
    """
    if not use_translator:
        print("⚠️ Translator not available, returning original text")
        return text
    
    # If already English, no translation needed
    if target_language.lower() == 'english':
        return text
    
    try:
        # Language code mapping
        language_codes = {
            'hindi': 'hi',
            'tamil': 'ta',
            'tamil (தமிழ்)': 'ta',
            'தமிழ்': 'ta',
            'telugu': 'te',
            'telugu (తెలుగు)': 'te',
            'తెలుగు': 'te',
            'marathi': 'mr',
            'bengali': 'bn',
            'gujarati': 'gu',
            'kannada': 'kn',
            'malayalam': 'ml',
            'punjabi': 'pa',
            'urdu': 'ur',
            'odia': 'or',
            'assamese': 'as',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'portuguese': 'pt',
            'russian': 'ru',
            'japanese': 'ja',
            'korean': 'ko',
            'chinese': 'zh-CN',
            'arabic': 'ar',
            'turkish': 'tr',
            'vietnamese': 'vi',
            'thai': 'th',
            'indonesian': 'id',
            'dutch': 'nl',
            'italian': 'it',
            'polish': 'pl',
            'ukrainian': 'uk',
            'persian': 'fa'
        }
        
        # Get language code
        lang_key = target_language.lower()
        lang_code = language_codes.get(lang_key, None)
        
        if not lang_code:
            print(f"⚠️ Unknown language: {target_language}, returning original")
            return text
        
        # Translate using deep-translator
        print(f"🌐 Translating to {target_language} ({lang_code})...")
        translator = GoogleTranslator(source='en', target=lang_code)
        translated_text = translator.translate(text)
        
        print(f"   Original: {text[:100]}...")
        print(f"   Translated: {translated_text[:100]}...")
        
        return translated_text
        
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        print(f"   Returning original text")
        return text

def get_gemini_response(user_message, mode, language='English'):
    """Get response from Gemini AI with Google Translate for non-English languages"""
    global gemini_base_model_name
    
    try:
        # Retrieve relevant context from RAG database (knowledge base)
        retrieved_context = retrieve_context(user_message, top_k=5)
        
        # Build context with RAG information
        rag_context = ""
        if retrieved_context:
            rag_context = f"\n\nKnowledge base context:\n{retrieved_context}\n"
        
        # ALWAYS generate in English for best quality with Gemma
        # Then use Google Translate if needed
        # Add crisis awareness to prompt for better safety
        crisis_context = ""
        if mode == 'professional':
            crisis_context = """
IMPORTANT CRISIS PROTOCOL:
If the user expresses suicidal thoughts or immediate danger, respond with deep empathy and provide these INDIAN crisis helplines:

🇮🇳 India Crisis Helplines (Available 24/7):
• Tele MANAS: 14416 or 1800-89-14416 (Available in 20 languages)
• KIRAN Mental Health: 1800-599-0019 (Free)
• Vandrevala Foundation: 1860-2662-345
• iCall: 9152987821

Do NOT provide generic advice for crisis situations. Always include these helplines."""
        
        prompt = f"""You are Aura, a compassionate mental health support chatbot for Indian users.
Provide warm, culturally-sensitive, supportive responses. Use emojis occasionally. Keep responses concise (2-4 sentences).{crisis_context}
{rag_context}
User message: {user_message}

Your response:"""
        
        # Use base model without system instructions (Gemma doesn't support it)
        language_model = genai.GenerativeModel(model_name=gemini_base_model_name)
        
        # Generate response in English
        response = language_model.generate_content(prompt)
        english_response = response.text.strip()
        
        print(f"🤖 Gemma response (English): {english_response[:100]}...")
        
        # Translate to target language if not English
        if language.lower() != 'english' and use_translator:
            print(f"🌐 Translating to {language} using Google Translate...")
            translated_response = translate_to_language(english_response, language)
            print(f"✅ Translation complete: {translated_response[:100]}...")
            return translated_response
        elif language.lower() != 'english' and not use_translator:
            print(f"⚠️ Translator not available, returning English response")
            return english_response
        else:
            # English - no translation needed
            return english_response
        
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        # Fallback to trained model
        return chatbot.get_response(user_message, mode=mode)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'trained_model_loaded': chatbot.vectorizer is not None,
        'gemini_enabled': use_gemini,
        'rag_enabled': use_rag,
        'ai_provider': 'gemini_with_rag' if (use_gemini and use_rag) else ('gemini' if use_gemini else 'trained_model')
    })

@app.route('/api/modes', methods=['GET'])
def get_modes():
    """Get available chat modes"""
    return jsonify({
        'modes': [
            {
                'id': 'friend',
                'name': 'Friend Mode',
                'description': 'Casual, supportive conversations'
            },
            {
                'id': 'professional',
                'name': 'Professional Mode',
                'description': 'Professional therapeutic guidance'
            }
        ]
    })

@app.route('/api/tts', methods=['POST', 'OPTIONS'])
def text_to_speech():
    """Generate speech audio from text using Google TTS"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')  # Language code (e.g., 'en', 'ta', 'hi')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Map frontend language codes to gTTS codes
        language_map = {
            'en-US': 'en',
            'hi-IN': 'hi',
            'ta-IN': 'ta',
            'te-IN': 'te',
            'mr-IN': 'mr',
            'bn-IN': 'bn',
            'gu-IN': 'gu',
            'kn-IN': 'kn',
            'ml-IN': 'ml',
            'pa-IN': 'pa',
            'ur-IN': 'ur',
            'es-ES': 'es',
            'fr-FR': 'fr',
            'de-DE': 'de',
            'pt-PT': 'pt',
            'ru-RU': 'ru',
            'ja-JP': 'ja',
            'ko-KR': 'ko',
            'zh-CN': 'zh-CN',
            'ar-SA': 'ar',
            'tr-TR': 'tr',
            'vi-VN': 'vi',
            'th-TH': 'th',
            'id-ID': 'id',
            'nl-NL': 'nl',
            'it-IT': 'it',
            'pl-PL': 'pl',
            'uk-UA': 'uk'
        }
        
        tts_lang = language_map.get(language, language.split('-')[0])
        
        # Generate speech
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        # Save to memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        print(f"🔊 Generated TTS audio for text: '{text[:50]}...' in language: {tts_lang}")
        
        return send_file(
            audio_buffer,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='speech.mp3'
        )
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AURA Backend Server Starting...")
    print("="*60)
    print(f"📍 Running on: http://127.0.0.1:5000")
    print(f"🎤 Voice input ready!")
    print(f"🤖 Using Gemini AI: {'✅' if use_gemini else '❌'}")
    print("="*60 + "\n")
    
    try:
        app.run(debug=False, port=5000, host='127.0.0.1', threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
