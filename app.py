# app.py - Main Flask Application
import os
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    """Analyzes text content and extracts key concepts"""
    
    def __init__(self):
        self.concept_map = {}
        self.chunks = []
        self.definitions = []
        self.processes = []
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove headers/footers (simple pattern matching)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'CHAPTER \d+', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into logical chunks based on structure"""
        chunks = []
        
        # Split by headings (lines ending with colon or in ALL CAPS)
        lines = text.split('\n')
        current_chunk = []
        current_heading = "Introduction"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a heading
            if (line.endswith(':') or 
                (line.isupper() and len(line.split()) <= 6) or
                re.match(r'^(#|\d+\.\s+|•)', line)):
                
                # Save previous chunk
                if current_chunk:
                    chunks.append({
                        'heading': current_heading,
                        'content': ' '.join(current_chunk),
                        'concepts': []
                    })
                    current_chunk = []
                
                current_heading = line.strip(':•# ')
            else:
                current_chunk.append(line)
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                'heading': current_heading,
                'content': ' '.join(current_chunk),
                'concepts': []
            })
        
        # If no chunks found, create one large chunk
        if not chunks:
            chunks.append({
                'heading': 'Main Content',
                'content': text,
                'concepts': []
            })
        
        self.chunks = chunks
        return chunks
    
    def extract_concepts(self, chunks: List[Dict]) -> Dict:
        """Extract key concepts from text chunks"""
        concept_map = defaultdict(list)
        
        # Patterns for concept extraction
        definition_patterns = [
            r'is (?:defined as|the|a|an) (.+?)[\.\n]',
            r'refers to (.+?)[\.\n]',
            r'known as (.+?)[\.\n]',
            r'called (.+?)[\.\n]'
        ]
        
        process_patterns = [
            r'process (?:of|for) (.+?)[\.\n]',
            r'steps (?:to|for) (.+?)[\.\n]',
            r'how to (.+?)[\.\n]',
            r'procedure for (.+?)[\.\n]'
        ]
        
        for chunk in chunks:
            content = chunk['content'].lower()
            
            # Extract definitions
            for pattern in definition_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    concept = match.split()[0] if match.split() else match
                    if len(concept) > 3 and concept not in ['the', 'a', 'an']:
                        concept_map[concept].append({
                            'type': 'definition',
                            'context': chunk['heading'],
                            'content': match[:100] + '...' if len(match) > 100 else match
                        })
            
            # Extract processes
            for pattern in process_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    concept = match.split()[0] if match.split() else match
                    if len(concept) > 3:
                        concept_map[concept].append({
                            'type': 'process',
                            'context': chunk['heading'],
                            'content': match[:100] + '...' if len(match) > 100 else match
                        })
            
            # Extract emphasized terms (capitalized or repeated)
            words = re.findall(r'\b[A-Z][a-z]+\b', chunk['content'])
            for word in words:
                if len(word) > 3 and word.lower() not in ['the', 'and', 'for', 'that']:
                    concept_map[word].append({
                        'type': 'term',
                        'context': chunk['heading'],
                        'content': 'Key term mentioned in text'
                    })
        
        # Keep only concepts mentioned multiple times or with definitions
        filtered_concepts = {}
        for concept, info in concept_map.items():
            if len(info) > 1 or any(item['type'] == 'definition' for item in info):
                filtered_concepts[concept] = info
        
        self.concept_map = filtered_concepts
        return filtered_concepts
    
    def analyze(self, text: str) -> Dict:
        """Main analysis function"""
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text.split()) < 50:
            raise ValueError("Text too short. Please provide at least 50 words.")
        
        chunks = self.chunk_text(cleaned_text)
        concepts = self.extract_concepts(chunks)
        
        return {
            'chunks': chunks,
            'concepts': concepts,
            'total_chunks': len(chunks),
            'total_concepts': len(concepts)
        }


class QuizGenerator:
    """Generates diagnostic quizzes based on text analysis"""
    
    def __init__(self):
        self.question_templates = {
            'conceptual': [
                "What is the main purpose or function of {concept} as described?",
                "How would you explain {concept} in your own words?",
                "What is the key difference between {concept} and similar ideas mentioned?",
                "Why is {concept} important in the context described?"
            ],
            'application': [
                "If you encountered a situation involving {concept}, how would you apply it?",
                "What would be a real-world example of {concept} based on the description?",
                "How might {concept} be used to solve a problem mentioned in the text?",
                "What would happen if {concept} was missing or not functioning?"
            ],
            'definition': [
                "Which definition best matches {concept} as described?",
                "What key characteristic defines {concept} according to the text?",
                "Based on the description, what is {concept} primarily concerned with?",
                "What is the scope or boundary of {concept} as mentioned?"
            ]
        }
    
    def generate_questions(self, analysis: Dict, num_questions: int = 10) -> List[Dict]:
        """Generate diagnostic questions from text analysis"""
        questions = []
        chunks = analysis['chunks']
        concepts = analysis['concepts']
        
        if not concepts:
            raise ValueError("No concepts found in text. Please provide more detailed content.")
        
        # Get top concepts (most mentioned or with definitions)
        concept_list = list(concepts.keys())
        
        # Question distribution
        question_types = ['conceptual'] * 4 + ['application'] * 3 + ['definition'] * 3
        
        for i in range(min(num_questions, len(concept_list))):
            concept = concept_list[i % len(concept_list)]
            q_type = question_types[i % len(question_types)]
            
            # Get template
            templates = self.question_templates[q_type]
            template = templates[i % len(templates)]
            
            # Generate question
            question_text = template.format(concept=concept)
            
            # Find relevant chunk
            chunk_info = concepts[concept][0] if concepts[concept] else {'context': 'Text'}
            source_chunk = next((c for c in chunks if c['heading'] == chunk_info['context']), chunks[0])
            
            # Generate options
            options = self.generate_options(concept, concepts, chunks, q_type)
            
            questions.append({
                'id': i + 1,
                'question': question_text,
                'type': q_type,
                'concept': concept,
                'options': options,
                'correct_answer': 0,  # First option is correct
                'source': source_chunk['heading'],
                'explanation': f"This concept is discussed in the section about {source_chunk['heading']}. The correct answer directly reflects the information provided in that section."
            })
        
        return questions
    
    def generate_options(self, concept: str, concepts: Dict, chunks: List[Dict], q_type: str) -> List[str]:
        """Generate multiple choice options"""
        correct = self.generate_correct_option(concept, concepts, q_type)
        
        # Generate distractors
        distractors = []
        
        # 1. Opposite of correct
        if 'not' in correct.lower() or 'never' in correct.lower():
            distractors.append(correct.replace('not ', '').replace('never ', 'always '))
        else:
            distractors.append(f"This is not accurate. {concept} works differently.")
        
        # 2. Misattributed property
        other_concepts = [c for c in concepts.keys() if c != concept]
        if other_concepts:
            random_concept = other_concepts[0]
            distractors.append(f"This actually describes {random_concept}, not {concept}.")
        
        # 3. Too broad/vague
        distractors.append(f"This is too general and doesn't specifically describe {concept}.")
        
        # 4. Completely wrong
        distractors.append(f"This contradicts the information about {concept} in the text.")
        
        # Combine and shuffle
        import random
        options = [correct] + random.sample(distractors, 3)
        random.shuffle(options)
        
        # Update correct answer index
        correct_index = options.index(correct)
        return options, correct_index
    
    def generate_correct_option(self, concept: str, concepts: Dict, q_type: str) -> str:
        """Generate correct answer based on concept type"""
        concept_info = concepts.get(concept, [{}])[0]
        
        if q_type == 'definition':
            if 'content' in concept_info:
                content = concept_info['content']
                # Extract simple definition
                if 'is' in content.lower():
                    return f"{concept} is {content.split('is', 1)[1].strip()}"
                return content
            return f"{concept} is a key concept discussed in the text."
        
        elif q_type == 'application':
            return f"{concept} can be applied in situations described in the text where similar principles are needed."
        
        else:  # conceptual
            return f"{concept} serves an important function as described, relating to other concepts in the material."


class LearnerAnalyzer:
    """Analyzes learner responses and generates personalized recommendations"""
    
    def __init__(self):
        self.learner_profiles = {
            'conceptual': {
                'description': 'Understands underlying principles and connections',
                'strengths': ['conceptual questions', 'making connections'],
                'weaknesses': ['memorization', 'specific details'],
                'techniques': ['Feynman Technique', 'Concept Mapping']
            },
            'rote': {
                'description': 'Good at memorizing facts and definitions',
                'strengths': ['definition questions', 'recall'],
                'weaknesses': ['application', 'critical thinking'],
                'techniques': ['Spaced Repetition', 'Active Recall']
            },
            'application': {
                'description': 'Excels at applying knowledge to practical situations',
                'strengths': ['application questions', 'problem-solving'],
                'weaknesses': ['theoretical understanding', 'foundations'],
                'techniques': ['Practice-based Learning', 'Case Studies']
            },
            'inconsistent': {
                'description': 'Variable performance across different types',
                'strengths': ['potential in multiple areas'],
                'weaknesses': ['consistency', 'identifying patterns'],
                'techniques': ['Interleaving', 'Mixed Practice']
            }
        }
    
    def analyze_responses(self, questions: List[Dict], answers: Dict, time_taken: int) -> Dict:
        """Analyze learner's quiz responses"""
        if len(answers) != len(questions):
            raise ValueError("Number of answers doesn't match number of questions")
        
        results = {
            'total_questions': len(questions),
            'correct_answers': 0,
            'incorrect_answers': 0,
            'accuracy': 0,
            'concept_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'type_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'time_per_question': time_taken / len(questions) if questions else 0
        }
        
        # Calculate performance
        for q in questions:
            q_id = str(q['id'])
            concept = q['concept']
            q_type = q['type']
            
            results['concept_performance'][concept]['total'] += 1
            results['type_performance'][q_type]['total'] += 1
            
            if q_id in answers and answers[q_id] == q['correct_answer']:
                results['correct_answers'] += 1
                results['concept_performance'][concept]['correct'] += 1
                results['type_performance'][q_type]['correct'] += 1
            else:
                results['incorrect_answers'] += 1
        
        results['accuracy'] = (results['correct_answers'] / results['total_questions']) * 100
        
        # Determine learner profile
        profile = self.determine_profile(results)
        
        # Identify strong and weak concepts
        strong_concepts, weak_concepts = self.identify_concepts(results)
        
        return {
            'results': results,
            'profile': profile,
            'strong_concepts': strong_concepts,
            'weak_concepts': weak_concepts,
            'recommendations': self.generate_recommendations(profile, strong_concepts, weak_concepts)
        }
    
    def determine_profile(self, results: Dict) -> str:
        """Determine learner's profile based on performance"""
        type_scores = {}
        for q_type, perf in results['type_performance'].items():
            if perf['total'] > 0:
                type_scores[q_type] = (perf['correct'] / perf['total']) * 100
        
        # Analyze pattern
        if not type_scores:
            return 'inconsistent'
        
        # Check if strong in conceptual but weak in definition
        conceptual_score = type_scores.get('conceptual', 0)
        definition_score = type_scores.get('definition', 0)
        application_score = type_scores.get('application', 0)
        
        if conceptual_score >= 70 and definition_score < 60:
            return 'conceptual'
        elif definition_score >= 70 and application_score < 60:
            return 'rote'
        elif application_score >= 70 and conceptual_score < 60:
            return 'application'
        else:
            # Check variance
            scores = list(type_scores.values())
            variance = max(scores) - min(scores) if scores else 0
            return 'inconsistent' if variance > 30 else 'balanced'
    
    def identify_concepts(self, results: Dict) -> Tuple[List[str], List[str]]:
        """Identify strong and weak concepts"""
        strong = []
        weak = []
        
        for concept, perf in results['concept_performance'].items():
            if perf['total'] > 0:
                accuracy = (perf['correct'] / perf['total']) * 100
                if accuracy >= 80:
                    strong.append(concept)
                elif accuracy <= 50:
                    weak.append(concept)
        
        return strong, weak
    
    def generate_recommendations(self, profile: str, strong: List[str], weak: List[str]) -> Dict:
        """Generate personalized study recommendations"""
        profile_info = self.learner_profiles.get(profile, self.learner_profiles['inconsistent'])
        
        # Study techniques
        techniques = profile_info['techniques']
        
        # Create study plan
        study_plan = []
        if weak:
            study_plan.append(f"Focus 60% of your study time on weak concepts: {', '.join(weak[:3])}")
        if strong:
            study_plan.append(f"Review strong concepts regularly: {', '.join(strong[:3])}")
        study_plan.append(f"Use the {techniques[0]} for deep learning")
        study_plan.append("Practice explaining concepts aloud")
        study_plan.append("Create summary notes after each study session")
        
        return {
            'profile_name': profile,
            'profile_description': profile_info['description'],
            'recommended_techniques': techniques,
            'why_fits': f"This approach matches your {profile} learning style by focusing on {profile_info['strengths'][0]} while addressing {profile_info['weaknesses'][0]}.",
            'study_plan': study_plan,
            'daily_time': "45-60 minutes per day",
            'focus_ratio': "60% weak topics, 30% medium topics, 10% strong topics",
            'what_to_avoid': [
                "Passive reading without testing yourself",
                "Cramming all studying into one session",
                "Ignoring your weak areas",
                "Studying without breaks"
            ]
        }


# Flask Application
from flask import Flask, render_template, request, jsonify, session
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Initialize components
text_analyzer = TextAnalyzer()
quiz_generator = QuizGenerator()
learner_analyzer = LearnerAnalyzer()

# Store analysis sessions
sessions = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze pasted text and generate quiz"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze text
        analysis = text_analyzer.analyze(text)
        
        # Generate quiz
        questions = quiz_generator.generate_questions(analysis)
        
        # Create session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'analysis': analysis,
            'questions': questions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Prepare response
        response = {
            'session_id': session_id,
            'analysis_summary': {
                'chunks_found': len(analysis['chunks']),
                'concepts_identified': len(analysis['concepts']),
                'top_concepts': list(analysis['concepts'].keys())[:5]
            },
            'quiz': questions,
            'total_questions': len(questions)
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/submit', methods=['POST'])
def submit_quiz():
    """Submit quiz answers and get analysis"""
    try:
        data = request.json
        session_id = data.get('session_id')
        answers = data.get('answers', {})
        time_taken = data.get('time_taken', 0)
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        session_data = sessions[session_id]
        questions = session_data['questions']
        
        # Analyze responses
        analysis = learner_analyzer.analyze_responses(questions, answers, time_taken)
        
        # Combine with session data
        result = {
            'quiz_results': {
                'accuracy': analysis['results']['accuracy'],
                'correct': analysis['results']['correct_answers'],
                'total': analysis['results']['total_questions']
            },
            'learner_profile': {
                'type': analysis['profile'],
                'description': analysis['recommendations']['profile_description']
            },
            'concept_analysis': {
                'strong': analysis['strong_concepts'][:5],
                'weak': analysis['weak_concepts'][:5]
            },
            'recommendations': analysis['recommendations']
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Submission error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/session/<session_id>')
def get_session(session_id):
    """Get session data"""
    if session_id in sessions:
        return jsonify(sessions[session_id])
    return jsonify({'error': 'Session not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
