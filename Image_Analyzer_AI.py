import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
import easyocr
from ultralytics import YOLO
from dotenv import load_dotenv
import spacy
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import json
import torch
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import google.generativeai as genai

class ImageAnalyzer:
    def __init__(self):
        """Initialize the Image Analyzer with all required models"""
        # Load environment variables
        load_dotenv()
        
        # Check and configure GPU
        self._setup_gpu()
        
        # Initialize InsightFace face recognition
        self.face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.threshold = 0.5
        self.player_img_dir = 'nba_player_images'
        self.db = self.build_database(self.player_img_dir)
        print(f"Database built for {len(self.db)} players.")
        
        # Initialize EasyOCR
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Initialize SpaCy NER model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Configure Gemini API key
        genai.configure(api_key=os.getenv("google_api_key"))
        
        # Keep track of conversation history for context
        self.conversation_history = []
        
        # Custom NER patterns for sports context
        self._setup_custom_ner()
    
    def _setup_gpu(self):
        """Setup GPU configuration for better performance"""
        print("\n[GPU CONFIGURATION]")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ CUDA GPU detected: {gpu_name}")
            print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
            print(f"✓ Using device: {device}")
            
            # Set device for YOLO
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"✓ MPS (Apple Silicon) detected")
            print(f"✓ Using device: {device}")
            
        else:
            device = torch.device("cpu")
            print("⚠ No GPU detected - using CPU")
            print("  Note: Performance will be slower. Consider:")
            print("  - Installing CUDA toolkit for NVIDIA GPUs")
            print("  - Using Apple Silicon Mac for MPS")
        
        self.device = device
    
    def _setup_custom_ner(self):
        """Setup custom NER patterns for sports context"""
        # Add custom patterns for team names, player names, etc.
        ruler = self.nlp.get_pipe("entity_ruler") if "entity_ruler" in self.nlp.pipe_names else self.nlp.add_pipe("entity_ruler")
        
        # Common basketball team patterns
        team_patterns = [
            {"label": "TEAM", "pattern": [{"LOWER": "lakers"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "celtics"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "warriors"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "bulls"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "heat"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "knicks"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "nets"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "clippers"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "spurs"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "mavericks"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "rockets"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "thunder"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "blazers"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "suns"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "kings"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "jazz"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "nuggets"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "timberwolves"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "pelicans"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "grizzlies"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "hawks"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "hornets"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "magic"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "wizards"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "pistons"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "cavaliers"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "pacers"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "indiana"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "bucks"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "raptors"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "76ers"}]},
            {"label": "TEAM", "pattern": [{"LOWER": "sixers"}]},
        ]
        
        ruler.add_patterns(team_patterns)
    
    def build_database(self, player_img_dir):
        db = {}
        for player in os.listdir(player_img_dir):
            player_dir = os.path.join(player_img_dir, player)
            if not os.path.isdir(player_dir):
                continue
            embeddings = []
            for img_file in os.listdir(player_dir):
                img_path = os.path.join(player_dir, img_file)
                img = cv2.imread(img_path)
                faces = self.face_app.get(img)
                if len(faces) == 1:
                    embeddings.append(faces[0].embedding)
            if embeddings:
                db[player] = embeddings
        return db
    
    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        """Detect faces and recognize NBA players using InsightFace"""
        print("\n[FACE RECOGNITION DETECTION]")
        img = cv2.imread(image_path)
        faces = self.face_app.get(img)
        results = []
        for face in faces:
            best_match = None
            best_score = 1.0
            for player, embeddings in self.db.items():
                for emb in embeddings:
                    score = cosine(face.embedding, emb)
                    if score < best_score:
                        best_score = score
                        best_match = player
            confidence = 1 - best_score
            if best_score < self.threshold and confidence >= 0.4:
                results.append({'bbox': face.bbox, 'name': best_match, 'confidence': confidence})
            elif best_score < self.threshold and confidence < 0.4:
                continue  # Ignore low-confidence matches
            elif best_score >= self.threshold and confidence >= 0.4:
                results.append({'bbox': face.bbox, 'name': 'Unknown', 'confidence': confidence})
            # else: ignore low-confidence unknowns
        detected_objects = {
            'people_count': len(results),
            'object_classes': [r['name'] for r in results],
            'boxes': [r['bbox'] for r in results],
            'results': results
        }
        print(f"Detected {detected_objects['people_count']} faces")
        print(f"Recognized: {detected_objects['object_classes']}")
        return detected_objects
    
    def extract_text_with_ocr(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract text from image using EasyOCR"""
        print("\n[OCR TEXT DETECTION]")
        ocr_results = self.ocr_reader.readtext(image_path)
        
        # Filter low-confidence results and combine nearby text
        extracted_texts = self._filter_and_combine_text(ocr_results)
        
        # Print results
        for text_info in extracted_texts:
            print(f"- Text: '{text_info['text']}' (Confidence: {text_info['confidence']:.2f})")
        
        return extracted_texts
    
    def analyze_text_with_ner(self, texts: List[str]) -> Dict[str, List[str]]:
        """Analyze extracted text using SpaCy NER"""
        print("\n[SPACY NER ANALYSIS]")
        
        # Combine all texts for analysis
        combined_text = " ".join(texts)
        doc = self.nlp(combined_text)
        
        # Extract different entity types
        entities = {
            'PERSON': [],
            'ORG': [],
            'TEAM': [],
            'GPE': [],  # Geographical entities (cities, countries)
            'CARDINAL': [],  # Numbers
            'MISC': []  # Miscellaneous entities
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
            else:
                entities['MISC'].append(ent.text)
        
        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        # Print results
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"{entity_type}: {', '.join(entity_list)}")
        
        # Special handling for team names that might be split
        self._enhance_team_recognition(texts, entities)
        
        return entities
    
    def _enhance_team_recognition(self, texts: List[str], entities: Dict[str, List[str]]):
        """Enhance team recognition by checking for partial matches"""
        # Common team name variations and partial matches
        team_variations = {
            'thunder': ['thunder', 'thun', 'der', 'hil', 'hilder'],
            'pacers': ['pacers', 'pacer', 'indiana'],
            'lakers': ['lakers', 'laker'],
            'celtics': ['celtics', 'celtic'],
            'warriors': ['warriors', 'warrior'],
            'bulls': ['bulls', 'bull'],
            'heat': ['heat'],
            'knicks': ['knicks', 'knick'],
            'nets': ['nets', 'net'],
            'clippers': ['clippers', 'clipper'],
            'spurs': ['spurs', 'spur'],
            'mavericks': ['mavericks', 'maverick'],
            'rockets': ['rockets', 'rocket'],
            'blazers': ['blazers', 'blazer'],
            'suns': ['suns', 'sun'],
            'kings': ['kings', 'king'],
            'jazz': ['jazz'],
            'nuggets': ['nuggets', 'nugget'],
            'timberwolves': ['timberwolves', 'timberwolf'],
            'pelicans': ['pelicans', 'pelican'],
            'grizzlies': ['grizzlies', 'grizzly'],
            'hawks': ['hawks', 'hawk'],
            'hornets': ['hornets', 'hornet'],
            'magic': ['magic'],
            'wizards': ['wizards', 'wizard'],
            'pistons': ['pistons', 'piston'],
            'cavaliers': ['cavaliers', 'cavalier'],
            'bucks': ['bucks', 'buck'],
            'raptors': ['raptors', 'raptor'],
            '76ers': ['76ers', 'sixers']
        }
        
        # Check each text for team name patterns
        for text in texts:
            text_lower = text.lower().strip()
            
            for team_name, variations in team_variations.items():
                for variation in variations:
                    if variation in text_lower:
                        if team_name not in entities['TEAM']:
                            entities['TEAM'].append(team_name.title())
                            print(f"Enhanced TEAM detection: {team_name.title()} (from '{text}')")
                        break
    
    def analyze_scene_with_llm(self, object_detection: Dict, ocr_texts: List[Dict], ner_entities: Dict, previous_dialog: str = None) -> str:
        """Analyze the scene using Gemini (google.generativeai) with all extracted information"""
        print("\n[LLM SCENE ANALYSIS]")
        
        # Prepare context for Gemini
        people_count = object_detection['people_count']
        all_texts = [item['text'] for item in ocr_texts]
        text_summary = " | ".join(all_texts)
        
        # Construct the prompt
        prompt = f"""
        You are two basketball analysts:
        - Alex, a play-by-play announcer.
        - Jamie, a former NBA player and color commentator.

        Current Scene Analysis:
        - People detected: {people_count}
        - Recognized players/people: {object_detection['object_classes']}
        - Teams/Organizations found in text: {', '.join(ner_entities['TEAM'])}, {', '.join(ner_entities['ORG'])}
        - Numbers/Scores found in text: {', '.join(ner_entities['CARDINAL'])}
        - Other text found on screen: {text_summary}
        """

        if previous_dialog:
            prompt += f"""
        Previously, the conversation was about: "{previous_dialog}"
        Continue the conversation naturally from there, incorporating the new scene information.
        """
        else:
            prompt += """
        This is the first frame. Start the conversation.
        """

        prompt += """
        Your Task:
        Have Alex and Jamie discuss the scene in a brief, back-and-forth conversation (2-3 exchanges each).
        - Interpret the data like a real analyst. Do not just list the raw data.
        - Talk about the teams, score, time, players, and what might be happening.
        - Make the conversation natural and insightful.

        Output Format Requirement:
        - Your entire output must be ONLY a raw JSON array of objects.
        - Do NOT wrap the JSON in markdown code blocks (i.e. no ```json).
        - Each object in the array must have two keys: "speaker" (string: "Alex" or "Jamie") and "utterance" (string).
        - Alternate speakers between each object.
        """
        
        print("Teams detected by NER:", ner_entities['TEAM'])
        print("\nPrompt sent to Gemini LLM:\n", prompt)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Clean up the response to ensure it's valid JSON.
        try:
            clean_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return clean_response
        except Exception:
            return response.text
    
    def analyze_image(self, image_path: str, previous_dialog: str = None) -> Dict[str, Any]:
        """Complete image analysis pipeline"""
        print(f"\n{'='*50}")
        print(f"ANALYZING IMAGE: {image_path}")
        print(f"{'='*50}")
        
        # Step 1: Object Detection
        object_detection = self.detect_objects(image_path)
        
        # Step 2: OCR Text Extraction
        ocr_texts = self.extract_text_with_ocr(image_path)
        
        # Step 3: NER Analysis
        if ocr_texts:
            all_texts = [item['text'] for item in ocr_texts]
            ner_entities = self.analyze_text_with_ner(all_texts)
        else:
            ner_entities = {'PERSON': [], 'ORG': [], 'TEAM': [], 'GPE': [], 'CARDINAL': [], 'MISC': []}
        
        # Step 4: LLM Scene Analysis
        scene_analysis = self.analyze_scene_with_llm(object_detection, ocr_texts, ner_entities, previous_dialog)
        
        # Step 5: Create enhanced visualization with both face recognition and OCR bounding boxes
        self._create_enhanced_visualization(image_path, object_detection, ocr_texts)
        
        # Return comprehensive results
        results = {
            'image_path': image_path,
            'object_detection': object_detection,
            'ocr_texts': ocr_texts,
            'ner_entities': ner_entities,
            'scene_analysis': scene_analysis
        }
        
        return results
    
    def _create_enhanced_visualization(self, image_path: str, object_detection: Dict, ocr_texts: List[Dict]):
        """Create enhanced visualization with both face recognition and OCR bounding boxes"""
        print("\n[CREATING ENHANCED VISUALIZATION]")
        
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image for visualization")
            return
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Create a copy for drawing
        vis_image = image.copy()
        
        # Draw face recognition boxes in GREEN
        if object_detection['people_count'] > 0:
            for i, (bbox, name, conf) in enumerate(
                [(r['bbox'], r['name'], r['confidence']) for r in object_detection['results']]
            ):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} ({conf:.2f})"
                cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw OCR text boxes in RED
        for i, text_info in enumerate(ocr_texts):
            bbox = text_info['bbox']
            text = text_info['text']
            confidence = text_info['confidence']
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"'{text}' ({confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "GREEN: Face Recognition", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, "RED: Text Detection (OCR)", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save the enhanced visualization
        output_path = "output_enhanced.jpg"
        cv2.imwrite(output_path, vis_image)
        print(f"✓ Enhanced visualization saved to: {output_path}")
    
    def _filter_and_combine_text(self, ocr_results: List[tuple]) -> List[Dict[str, Any]]:
        """Filter low-confidence text and combine nearby text fragments"""
        # Filter out low-confidence results (< 0.5)
        filtered_results = []
        for bbox, text, conf in ocr_results:
            if conf >= 0.5:
                filtered_results.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox,
                    'center': self._get_bbox_center(bbox)
                })
        
        # Combine nearby text fragments
        combined_results = self._combine_nearby_text(filtered_results)
        
        return combined_results
    
    def _get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return (center_x, center_y)
    
    def _combine_nearby_text(self, text_results: List[Dict]) -> List[Dict]:
        """Combine text fragments that are close to each other"""
        if len(text_results) <= 1:
            return text_results
        
        # Sort by x-coordinate (left to right)
        sorted_results = sorted(text_results, key=lambda x: x['center'][0])
        
        combined_results = []
        i = 0
        
        while i < len(sorted_results):
            current = sorted_results[i]
            combined_text = current['text']
            combined_bbox = current['bbox']
            combined_conf = current['confidence']
            
            # Look for nearby text to combine
            j = i + 1
            while j < len(sorted_results):
                next_text = sorted_results[j]
                
                # Check if texts are close horizontally (within 50 pixels)
                distance = abs(current['center'][0] - next_text['center'][0])
                if distance < 50:
                    # Combine text
                    combined_text += " " + next_text['text']
                    combined_conf = min(combined_conf, next_text['confidence'])  # Take lower confidence
                    
                    # Expand bounding box to include both
                    combined_bbox = self._expand_bbox(combined_bbox, next_text['bbox'])
                    j += 1
                else:
                    break
            
            # Add combined result
            combined_results.append({
                'text': combined_text,
                'confidence': combined_conf,
                'bbox': combined_bbox,
                'center': self._get_bbox_center(combined_bbox)
            })
            
            i = j
        
        return combined_results
    
    def _expand_bbox(self, bbox1, bbox2):
        """Expand bounding box to include both bboxes"""
        x_coords = [point[0] for point in bbox1 + bbox2]
        y_coords = [point[1] for point in bbox1 + bbox2]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

def main():
    """Main function to run the image analyzer"""
    analyzer = ImageAnalyzer()
    
    # Replace with your actual image path
    image_path = "test_nba_photo.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Run complete analysis
    results = analyzer.analyze_image(image_path)
    
    # Print final summary
    print(f"\n{'='*50}")
    print("FINAL ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(results['scene_analysis'])
    
    # Save results to JSON for future reference
    with open('analysis_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to 'analysis_results.json'")
    print(f"Enhanced visualization saved to 'output_enhanced.jpg'")
    print(f"\n📊 VISUALIZATION FILES:")
    print(f"  • output_enhanced.jpg - Shows both face recognition and text (RED) boxes")
    print(f"  • analysis_results.json - Detailed analysis data")

if __name__ == "__main__":
    main() 
