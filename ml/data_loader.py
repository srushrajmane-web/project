"""
Data Generator & Dataset Setup for Human Disease Diagnosis System.
Creates realistic medical datasets for:
1. Multi-class Symptom-Disease Dataset (42 Diseases, 132 Symptoms)
2. Disease Descriptions, Precautions, Medications, Diet, and Workout CSVs
3. Pima Indians Diabetes Dataset
4. UCI Cleveland Heart Disease Dataset
5. UCI Parkinson's Disease Biomedical Voice Dataset
"""

import os
import csv
import json
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 42 Diseases and their characteristic symptoms catalogue
DISEASES_DATA = {
    "Fungal infection": {
        "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches"],
        "category": "Dermatology",
        "description": "Fungal infection is a skin condition caused by a fungus. Common symptoms include itchy, scaly, or reddish skin patches.",
        "precautions": ["Bath twice daily", "Use dettol or antiseptic soap", "Keep infected area dry and clean", "Do not share personal items like towels"],
        "medications": ["Antifungal creams (Clotrimazole, Terbinafine)", "Oral antifungals (Fluconazole)", "Topical powder"],
        "diet": ["Garlic and turmeric rich diet", "Probiotics and yogurt", "Reduce refined sugar intake", "Stay hydrated"],
        "workout": ["Light yoga", "Avoid intense workouts that cause excessive sweating", "Wear breathable cotton clothing"]
    },
    "Allergy": {
        "symptoms": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
        "category": "Immunology",
        "description": "An allergic reaction occurs when the immune system overreacts to a harmless substance such as pollen, dust, or pet dander.",
        "precautions": ["Avoid known allergy triggers", "Use air filters or purifiers", "Wear a mask outdoors during pollen season", "Keep windows closed during high pollen times"],
        "medications": ["Antihistamines (Cetirizine, Loratadine)", "Nasal corticosteroid sprays", "Decongestants"],
        "diet": ["Vitamin C rich citrus fruits", "Ginger and green tea", "Anti-inflammatory foods", "Omega-3 rich foods"],
        "workout": ["Indoor light aerobics", "Gentle stretching", "Avoid strenuous outdoor running on high pollen days"]
    },
    "GERD": {
        "symptoms": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
        "category": "Gastroenterology",
        "description": "Gastroesophageal reflux disease (GERD) occurs when stomach acid repeatedly flows back into the tube connecting your mouth and stomach.",
        "precautions": ["Avoid lying down immediately after meals", "Eat smaller and more frequent meals", "Avoid spicy, fatty, and acidic foods", "Elevate head of bed while sleeping"],
        "medications": ["Proton Pump Inhibitors (Omeprazole, Pantoprazole)", "H2 Blockers (Famotidine)", "Antacids"],
        "diet": ["Oatmeal, non-citrus fruits, lean poultry", "Ginger tea", "Alkaline foods like bananas", "Avoid caffeine and alcohol"],
        "workout": ["Low-impact walking", "Gentle cycling", "Avoid heavy weightlifting or exercises that compress abdomen"]
    },
    "Chronic cholestasis": {
        "symptoms": ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
        "category": "Hepatology",
        "description": "Chronic cholestasis is a liver condition where bile flow from the liver is reduced or blocked over a sustained duration.",
        "precautions": ["Consult a gastroenterologist/hepatologist", "Avoid alcohol completely", "Limit high-fat dietary intake", "Get periodic liver function tests"],
        "medications": ["Ursodeoxycholic acid (UDCA)", "Fat-soluble vitamin supplements (A, D, E, K)", "Antipruritic agents"],
        "diet": ["Low-fat balanced diet", "High-fiber vegetables", "Lean proteins", "Adequate hydration"],
        "workout": ["Brisk walking", "Low-intensity aerobics", "Adequate rest between activities"]
    },
    "Drug Reaction": {
        "symptoms": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting_ urination"],
        "category": "Allergy & Clinical Immunology",
        "description": "An adverse drug reaction is an unwanted or unexpected effect caused by taking a medication, supplement, or prescription drug.",
        "precautions": ["Discontinue suspected offending medication immediately", "Consult prescribing physician", "Maintain list of known drug allergies", "Stay hydrated"],
        "medications": ["Antihistamines", "Topical soothing lotions (Calamine)", "Oral corticosteroids if prescribed"],
        "diet": ["Plenty of water and electrolyte fluids", "Bland diet (BRAT diet)", "Fresh fruits", "Avoid processed foods"],
        "workout": ["Rest and avoid vigorous exertion", "Light walking when recovered", "Avoid heat exposure"]
    },
    "Peptic ulcer disease": {
        "symptoms": ["vomiting", "indigestion", "loss_of_appetite", "abdominal_pain", "passage_of_gases", "internal_itching"],
        "category": "Gastroenterology",
        "description": "Peptic ulcers are open sores that develop on the inside lining of your stomach and the upper portion of your small intestine.",
        "precautions": ["Avoid NSAID painkillers without prescription", "Limit spicy, acidic foods and alcohol", "Do not smoke", "Eat meals on a regular schedule"],
        "medications": ["Antibiotics for H. pylori eradication", "Proton Pump Inhibitors", "Sucralfate mucosal protectants"],
        "diet": ["Probiotic foods like kefir and yogurt", "High fiber foods like apples and carrots", "Cabbage juice and leafy greens", "Avoid fried and acidic foods"],
        "workout": ["Mild walking", "Stress-reducing meditation and deep breathing", "Gentle yoga poses"]
    },
    "AIDS": {
        "symptoms": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts"],
        "category": "Infectious Disease",
        "description": "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV).",
        "precautions": ["Strict adherence to Antiretroviral Therapy (ART)", "Practice safe barrier protection", "Screen for opportunistic infections", "Avoid unpasteurized food"],
        "medications": ["Antiretroviral therapy (Tenofovir, Emtricitabine, Dolutegravir)", "Prophylactic antibiotics"],
        "diet": ["High-protein nutrient-dense diet", "Clean thoroughly cooked meals", "Vitamin and mineral supplements", "Safe treated drinking water"],
        "workout": ["Moderate resistance training to prevent muscle wasting", "Low-impact cardio", "Gentle stretching"]
    },
    "Diabetes ": {
        "symptoms": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "obesity", "excessive_hunger", "increased_appetite", "polyuria"],
        "category": "Endocrinology",
        "description": "Diabetes mellitus is a metabolic disease characterized by chronic high blood glucose levels due to insulin deficiency or resistance.",
        "precautions": ["Monitor blood glucose levels regularly", "Maintain a balanced low-glycemic diet", "Take prescribed hypoglycemic medications or insulin", "Inspect feet daily for sores or cuts"],
        "medications": ["Metformin", "SGLT2 inhibitors (Empagliflozin)", "GLP-1 receptor agonists", "Insulin therapy when indicated"],
        "diet": ["Low glycemic index foods", "Whole grains, legumes, green leafy vegetables", "Lean proteins", "Avoid sugary sodas and sweets"],
        "workout": ["30 minutes daily brisk walking", "Moderate resistance training twice weekly", "Swimming or cycling"]
    },
    "Gastroenteritis": {
        "symptoms": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
        "category": "Gastroenterology",
        "description": "Gastroenteritis is an inflammation of the lining of the intestines caused by a virus, bacteria, or parasites (stomach flu).",
        "precautions": ["Drink oral rehydration solutions (ORS)", "Wash hands frequently with soap", "Eat small bland meals", "Seek urgent care if blood in stool or high fever"],
        "medications": ["Oral Rehydration Salts (ORS)", "Antiemetics if prescribed (Ondansetron)", "Zinc supplements", "Probiotics"],
        "diet": ["Bananas, rice, applesauce, toast (BRAT diet)", "Clear broths and coconut water", "Electrolyte fluids", "Avoid dairy and fatty foods"],
        "workout": ["Complete physical rest until fully rehydrated", "Gentle short walks once recovered"]
    },
    "Bronchial Asthma": {
        "symptoms": ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
        "category": "Pulmonology",
        "description": "Asthma is a chronic condition in which your airways narrow and swell and may produce extra mucus, causing difficulty breathing.",
        "precautions": ["Always carry rescue inhaler (Albuterol)", "Avoid dust, cold air, smoke, and pet dander", "Take maintenance controller inhalers daily", "Use a peak flow meter to track lung function"],
        "medications": ["Inhaled Corticosteroids (Budesonide, Fluticasone)", "Short-acting Beta Agonists (Albuterol)", "Leukotriene receptor antagonists (Montelukast)"],
        "diet": ["Foods rich in Vitamin D and Magnesium", "Omega-3 fatty acids", "Fresh berries and spinach", "Avoid sulfites in dried fruits and wine"],
        "workout": ["Swimming in warm humid environments", "Indoor walking", "Pranayama and diaphragmatic breathing exercises"]
    },
    "Hypertension ": {
        "symptoms": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
        "category": "Cardiology",
        "description": "Hypertension (high blood pressure) is a common condition where the long-term force of the blood against artery walls is elevated.",
        "precautions": ["Measure blood pressure regularly", "Reduce dietary sodium (< 2,300 mg/day)", "Manage stress and get 7-8 hours sleep", "Limit alcohol and quit smoking"],
        "medications": ["ACE Inhibitors (Lisinopril)", "ARBs (Losartan)", "Calcium Channel Blockers (Amlodipine)", "Thiazide Diuretics"],
        "diet": ["DASH Diet (Dietary Approaches to Stop Hypertension)", "Potassium-rich foods (bananas, sweet potatoes)", "Berries, beetroot juice, garlic", "Low sodium foods"],
        "workout": ["Aerobic exercise 150 mins/week (brisk walking, jogging)", "Light swimming", "Yoga and mindfulness meditation"]
    },
    "Migraine": {
        "symptoms": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "irritability", "visual_disturbances"],
        "category": "Neurology",
        "description": "A migraine is a neurological condition that causes severe throbbing pain or a pulsing sensation, usually on one side of the head, often with nausea and light sensitivity.",
        "precautions": ["Rest in a quiet, dark room during attacks", "Identify and avoid personal dietary triggers", "Maintain consistent sleep and meal schedules", "Stay well-hydrated throughout the day"],
        "medications": ["Triptans (Sumatriptan, Rizatriptan)", "NSAIDs (Naproxen, Ibuprofen)", "CGRP antagonists or beta-blockers for prevention"],
        "diet": ["Magnesium-rich leafy greens, pumpkin seeds", "Ginger tea", "Stay hydrated", "Avoid aged cheeses, chocolate, MSG, and artificial sweeteners"],
        "workout": ["Moderate steady walking", "Gentle neck and shoulder stretching", "Relaxation yoga during headache-free days"]
    },
    "Cervical spondylosis": {
        "symptoms": ["back_pain", "neck_pain", "dizziness", "loss_of_balance"],
        "category": "Orthopedics & Rheumatology",
        "description": "Cervical spondylosis is age-related wear and tear affecting the spinal disks in your neck, leading to stiffness and radiating pain.",
        "precautions": ["Maintain ergonomic posture while working", "Use an orthopedic cervical support pillow", "Perform prescribed neck isometric exercises", "Avoid lifting heavy weights on head or shoulders"],
        "medications": ["NSAIDs (Ibuprofen, Celecoxib)", "Muscle relaxants (Cyclobenzaprine)", "Topical analgesic gels"],
        "diet": ["Calcium and Vitamin D rich foods", "Turmeric and anti-inflammatory foods", "Bone broth and collagen sources", "Adequate hydration"],
        "workout": ["Cervical isometric strengthening exercises", "Gentle neck rotations and chin tucks", "Low-impact swimming"]
    },
    "Paralysis (brain hemorrhage)": {
        "symptoms": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
        "category": "Emergency Neurology / Stroke Care",
        "description": "Brain hemorrhage stroke occurs when a weakened vessel ruptures and bleeds into the brain, causing sudden paralysis or loss of function on one side.",
        "precautions": ["URGENT: Call emergency medical services (911/112) immediately", "Keep patient lying flat on side to prevent aspiration", "Do not give food, water, or aspirin", "Monitor breathing and level of consciousness"],
        "medications": ["Emergency hospital neuro-critical care", "Blood pressure management agents", "Anticonvulsants / osmotic agents if indicated"],
        "diet": ["Doctor-directed dysphagia nutrition", "Low-sodium pureed nutrient-rich meals", "Adequate fluid management"],
        "workout": ["Supervised post-stroke physical rehabilitation", "Occupational therapy", "Passive range-of-motion limb exercises"]
    },
    "Jaundice": {
        "symptoms": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain"],
        "category": "Hepatology",
        "description": "Jaundice is a yellow discoloration of the skin, mucous membranes, and sclera caused by elevated levels of bilirubin in the blood.",
        "precautions": ["Get comprehensive liver function and viral hepatitis panels", "Drink boiled and filtered water", "Avoid oily, fried, and processed foods", "Get complete physical rest"],
        "medications": ["Supportive therapy based on underlying cause", "Liver tonic formulations", "Fat-soluble vitamins"],
        "diet": ["Sugarcane juice, radishes, papaya", "High-carbohydrate easily digestible foods", "Barley water and coconut water", "Avoid fats and alcohol"],
        "workout": ["Strict bed rest during acute phase", "Gentle walking after bilirubin normalizes"]
    },
    "Malaria": {
        "symptoms": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain", "diarrhoea"],
        "category": "Infectious Disease",
        "description": "Malaria is a mosquito-borne infectious disease caused by Plasmodium parasites, characterized by cyclical high fevers, shaking chills, and sweating.",
        "precautions": ["Use insecticide-treated mosquito bed nets", "Apply mosquito repellents (DEET)", "Complete full course of antimalarial medications", "Eliminate stagnant water around home"],
        "medications": ["Artemisinin-based combination therapies (ACTs)", "Chloroquine or Artemether-Lumefantrine", "Antipyretics (Paracetamol)"],
        "diet": ["High calorie fluid diet", "Fresh orange and lemon juice", "Steamed vegetables and lentils", "Electrolyte fluids"],
        "workout": ["Complete rest during fever episodes", "Resume gentle movement only after parasitic clearance"]
    },
    "Chicken pox": {
        "symptoms": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body"],
        "category": "Pediatrics & Infectious Disease",
        "description": "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus, causing an itchy, blister-like rash all over the body.",
        "precautions": ["Isolate from non-immune individuals until all blisters crust", "Avoid scratching blisters to prevent bacterial infection and scarring", "Trim fingernails short", "Take cool colloidal oatmeal baths"],
        "medications": ["Antiviral therapy (Acyclovir) if high risk", "Calamine lotion", "Antihistamines for itching", "Paracetamol for fever (Avoid Aspirin)"],
        "diet": ["Cool soft foods, soups, smoothies", "Hydrating fresh fruits", "Avoid salty, spicy, or acidic foods that irritate mouth sores"],
        "workout": ["Rest at home in a cool room", "Avoid exertion until skin lesions are fully scabbed"]
    },
    "Dengue": {
        "symptoms": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain", "red_spots_over_body"],
        "category": "Infectious Disease",
        "description": "Dengue is a viral infection transmitted by Aedes mosquitoes, causing sudden high fever, retro-orbital eye pain, severe bone pain, and risk of thrombocytopenia.",
        "precautions": ["Monitor platelet count and hematocrit closely", "Avoid NSAIDs/Aspirin due to bleeding risk", "Drink plentiful fluids with electrolytes", "Seek immediate ER care if persistent vomiting or bleeding occurs"],
        "medications": ["Paracetamol for fever/pain relief", "IV fluids for severe cases", "Platelet transfusion if indicated"],
        "diet": ["Papaya leaf extract / kiwi", "Pomegranate juice and coconut water", "High protein lentil soups", "Abundant fluids"],
        "workout": ["Absolute bed rest during acute febrile and critical phases", "Gradual recovery over 2-3 weeks"]
    },
    "Typhoid": {
        "symptoms": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look_(typhos)", "belly_pain"],
        "category": "Infectious Disease",
        "description": "Typhoid fever is a life-threatening systemic infection caused by Salmonella Typhi bacteria, spread through contaminated food and water.",
        "precautions": ["Drink only boiled or bottled water", "Wash hands thoroughly before eating", "Complete full antibiotic course prescribed by physician", "Eat only freshly prepared and hot cooked food"],
        "medications": ["Antibiotics (Ceftriaxone, Azithromycin, Ciprofloxacin)", "Antipyretics for fever"],
        "diet": ["High-calorie soft bland diet (rice porridge, boiled potatoes)", "Bananas and applesauce", "Clear broths", "Avoid spicy, fibrous, or raw foods"],
        "workout": ["Complete rest until blood cultures are negative and fever resolves completely"]
    },
    "hepatitis A": {
        "symptoms": ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes", "muscle_pain"],
        "category": "Hepatology & Infectious Disease",
        "description": "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus (HAV), usually transmitted via contaminated food or water.",
        "precautions": ["Practice strict hand hygiene", "Avoid alcohol and hepatotoxic medications", "Vaccinate close family contacts", "Consume boiled water and peel all fruits"],
        "medications": ["Supportive care and hydration", "Antiemetics if needed", "Liver support therapy"],
        "diet": ["High-carbohydrate easily digestible diet", "Fruit juices and vegetable broths", "Low fat, high lean protein", "Plenty of clean water"],
        "workout": ["Adequate bed rest during acute jaundice phase"]
    },
    "Hepatitis B": {
        "symptoms": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "malaise", "receiving_blood_transfusion", "receiving_unsterile_injections"],
        "category": "Hepatology",
        "description": "Hepatitis B is a serious liver infection caused by HBV that can become chronic and lead to cirrhosis or liver cancer.",
        "precautions": ["Consult a hepatologist for viral load and liver staging", "Never share razors, needles, or toothbrushes", "Vaccinate sexual partners and household members", "Completely abstain from alcohol"],
        "medications": ["Antiviral medications (Tenofovir, Entecavir)", "Pegylated interferon"],
        "diet": ["Antioxidant-rich Mediterranean diet", "Whole grains, berries, and cruciferous vegetables", "Lean proteins", "Avoid raw shellfish and alcohol"],
        "workout": ["Moderate aerobic exercises like walking or swimming 30 mins/day"]
    },
    "Hepatitis C": {
        "symptoms": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes", "family_history"],
        "category": "Hepatology",
        "description": "Hepatitis C is a blood-borne viral infection causing chronic liver inflammation, now largely curable with direct-acting antivirals.",
        "precautions": ["Complete full course of direct-acting antiviral (DAA) therapy", "Regular screening for liver fibrosis and hepatoma", "Avoid alcohol completely", "Avoid sharing personal hygiene items"],
        "medications": ["Direct-acting antivirals (Sofosbuvir, Velpatasvir, Glecaprevir)"],
        "diet": ["Liver-healthy plant-based diet", "Green tea and antioxidant foods", "Low sodium to prevent fluid retention"],
        "workout": ["Regular light to moderate cardiovascular workouts"]
    },
    "Hepatitis D": {
        "symptoms": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
        "category": "Hepatology",
        "description": "Hepatitis D (delta hepatitis) is a liver infection caused by HDV that only occurs in people infected with Hepatitis B.",
        "precautions": ["Specialist hepatology co-management", "Prevention of Hepatitis B transmission", "Periodic liver elastography/ultrasound", "Avoid alcohol and hepatotoxic drugs"],
        "medications": ["Pegylated interferon alfa", "Bulevirtide (where approved)", "Supportive liver therapies"],
        "diet": ["High protein, easily digestible balanced diet", "Fresh organic vegetables and fruits"],
        "workout": ["Gentle walking, avoid heavy exhaustion"]
    },
    "Hepatitis E": {
        "symptoms": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure"],
        "category": "Hepatology",
        "description": "Hepatitis E is a waterborne liver disease caused by HEV, particularly hazardous in pregnant women.",
        "precautions": ["Ensure strict water sanitation and boiling", "Avoid undercooked pork or game meats", "Immediate hospitalization if pregnant or developing encephalopathy", "Adequate rest"],
        "medications": ["Supportive clinical care", "Ribavirin in select chronic immunocompromised cases"],
        "diet": ["Light carbohydrate-rich diet", "Fresh juices, coconut water, boiled vegetables"],
        "workout": ["Bed rest during the acute phase"]
    },
    "Alcoholic hepatitis": {
        "symptoms": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload"],
        "category": "Hepatology & Addiction Medicine",
        "description": "Alcoholic hepatitis is severe liver inflammation caused by heavy, prolonged alcohol consumption.",
        "precautions": ["Permanent and complete cessation of alcohol", "Seek professional addiction counseling support", "Monitor for ascites, encephalopathy, and GI bleeding", "Regular liver function monitoring"],
        "medications": ["Corticosteroids (Prednisolone) in severe cases", "Nutritional supplements and Thiamine (Vitamin B1)", "Pentoxifylline"],
        "diet": ["High calorie, high protein nutrition", "Sodium restricted (< 2g/day) to manage fluid swelling", "Frequent small nutrient-dense meals"],
        "workout": ["Mild physical therapy and restorative walking as tolerated"]
    },
    "Tuberculosis": {
        "symptoms": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "phlegm", "chest_pain", "blood_in_sputum"],
        "category": "Pulmonology & Infectious Disease",
        "description": "Tuberculosis (TB) is a serious bacterial disease caused by Mycobacterium tuberculosis that primarily affects the lungs.",
        "precautions": ["Adhere strictly to 6-month Directly Observed Treatment (DOTS)", "Wear a respirator mask when around others during contagious phase", "Ensure good room ventilation and sunlight", "Never skip doses to avoid drug resistance"],
        "medications": ["First-line quadruple therapy: Isoniazid, Rifampicin, Pyrazinamide, Ethambutol", "Pyridoxine (Vitamin B6)"],
        "diet": ["High calorie, high protein diet (eggs, milk, legumes)", "Vitamin A, C, E, and Zinc supplements", "Frequent nutritious meals"],
        "workout": ["Deep breathing exercises", "Gentle walks once sputum conversion occurs"]
    },
    "Common Cold": {
        "symptoms": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "swelled_lymph_nodes", "malaise", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "loss_of_smell"],
        "category": "Primary Care / ENT",
        "description": "The common cold is a viral infection of your upper respiratory tract, most commonly caused by rhinoviruses.",
        "precautions": ["Get plenty of rest and sleep", "Stay well hydrated with warm liquids", "Wash hands regularly to avoid spreading", "Use steam inhalation for nasal congestion"],
        "medications": ["Decongestants (Pseudoephedrine)", "Saline nasal rinse", "Paracetamol or Ibuprofen", "Throat lozenges"],
        "diet": ["Warm chicken soup or vegetable broth", "Hot herbal tea with honey and lemon", "Citrus fruits", "Ginger and garlic"],
        "workout": ["Rest until fever and body aches subside; light stretching only"]
    },
    "Pneumonia": {
        "symptoms": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
        "category": "Pulmonology",
        "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or purulent material.",
        "precautions": ["Complete the entire prescribed antibiotic/antiviral course", "Use a pulse oximeter to track oxygen saturation (>94%)", "Perform incentive spirometry lung expansion", "Seek urgent care if experiencing severe shortness of breath or blue lips"],
        "medications": ["Antibiotics (Amoxicillin-clavulanate, Azithromycin, Levofloxacin)", "Bronchodilators", "Antipyretics"],
        "diet": ["Protein-rich soups and broths", "Plenty of warm fluids", "Foods rich in vitamins C and A"],
        "workout": ["Incentive spirometry and deep diaphragmatic breathing", "Gentle walking when afebrile"]
    },
    "Dimorphic hemmorhoids(piles)": {
        "symptoms": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
        "category": "Proctology & Colorectal Surgery",
        "description": "Hemorrhoids are swollen veins in your anus and lower rectum, which can cause bleeding, pain, and itching.",
        "precautions": ["Avoid straining during bowel movements", "Take warm sitz baths for 15 minutes twice daily", "Do not delay urge to defecate", "Avoid prolonged sitting on toilet"],
        "medications": ["Topical hydrocortisone / lidocaine creams", "Stool softeners (Docusate sodium)", "Fiber supplements (Psyllium husk)"],
        "diet": ["High-fiber diet (beans, whole grains, broccoli, berries)", "Drink 8-10 glasses of water daily", "Prunes and flaxseeds"],
        "workout": ["Regular walking to promote peristalsis", "Pelvic floor Kegel exercises", "Avoid heavy squats and deadlifts"]
    },
    "Heart attack": {
        "symptoms": ["vomiting", "breathlessness", "sweating", "chest_pain"],
        "category": "Emergency Cardiology",
        "description": "A heart attack (myocardial infarction) is a medical emergency where blood flow to a part of the heart muscle is blocked.",
        "precautions": ["CALL 911 / 112 IMMEDIATELY", "Chew an adult aspirin (325 mg) if not allergic", "Rest in a comfortable semi-seated position", "Do not drive yourself to the hospital"],
        "medications": ["Emergency hospital reperfusion (Angioplasty/Stenting)", "Antiplatelets (Aspirin, Clopidogrel)", "Beta-blockers, ACE inhibitors, Statins"],
        "diet": ["Strict low-sodium, heart-healthy Mediterranean diet", "Zero trans-fats", "Omega-3 rich fish and olive oil"],
        "workout": ["Supervised phase II cardiac rehabilitation only"]
    },
    "Varicose veins": {
        "symptoms": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
        "category": "Vascular Surgery",
        "description": "Varicose veins are enlarged, twisted veins that usually occur in the legs when vein valves become weakened or faulty.",
        "precautions": ["Wear graduated compression stockings daily", "Elevate legs above heart level when resting", "Avoid standing or sitting in one position for long periods", "Maintain a healthy body weight"],
        "medications": ["Flavonoid venoactive agents", "Topical soothing gels", "Sclerotherapy / laser ablation if advised by vascular surgeon"],
        "diet": ["High-fiber, low-sodium foods", "Flavonoid-rich foods (berries, citrus, bell peppers)", "Adequate water intake"],
        "workout": ["Calf raises and ankle pumps", "Low-impact walking and swimming", "Bicycling"]
    },
    "Hypothyroidism": {
        "symptoms": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "depression", "irritability", "abnormal_menstruation"],
        "category": "Endocrinology",
        "description": "Hypothyroidism (underactive thyroid) is a condition where the thyroid gland does not produce enough crucial thyroid hormones.",
        "precautions": ["Take thyroid hormone replacement on an empty stomach in morning", "Have periodic TSH and Free T4 blood tests every 6-12 months", "Maintain consistent iodine intake", "Stay active to boost metabolism"],
        "medications": ["Levothyroxine (Synthroid / Eltroxin)"],
        "diet": ["Selenium and zinc rich foods (Brazil nuts, pumpkin seeds)", "Iodized salt in moderation", "Whole vegetables and lean proteins", "Limit raw goitrogens (raw cabbage/broccoli)"],
        "workout": ["Aerobic exercise 30 mins 4x/week", "Strength training to improve metabolic rate", "Yoga for vitality"]
    },
    "Hyperthyroidism": {
        "symptoms": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger", "muscle_weakness", "irritability", "abnormal_menstruation"],
        "category": "Endocrinology",
        "description": "Hyperthyroidism (overactive thyroid) is the overproduction of thyroid hormones, accelerating the body's metabolism.",
        "precautions": ["Take antithyroid medications consistently as prescribed", "Monitor resting heart rate and blood pressure", "Avoid excess dietary iodine and kelp supplements", "Protect eyes with sunglasses if experiencing proptosis"],
        "medications": ["Antithyroid drugs (Methimazole, Propylthiouracil)", "Beta-blockers (Propranolol) for palpitations"],
        "diet": ["Low-iodine diet if recommended", "Calcium and Vitamin D to protect bone density", "Cruciferous vegetables (cabbage, broccoli)", "Avoid caffeine and energy drinks"],
        "workout": ["Low-impact gentle walking", "Restorative yoga and meditation", "Avoid high-intensity exhaustive workouts"]
    },
    "Hypoglycemia": {
        "symptoms": ["vomiting", "headache", "nausea", "sweating", "anxiety", "slurred_speech", "irritability", "excessive_hunger", "drying_and_tingling_lips", "palpitations"],
        "category": "Endocrinology / Emergency Medicine",
        "description": "Hypoglycemia is a condition characterized by abnormally low blood glucose (sugar) levels (<70 mg/dL).",
        "precautions": ["Follow the 15-15 Rule: consume 15g fast carbs, wait 15 mins, re-test", "Always carry glucose tablets or candy", "Never skip or delay meals after taking insulin/secretagogues", "Wear a medical alert bracelet"],
        "medications": ["Oral glucose tablets / gels", "Glucagon emergency kit (nasal Baqsimi or injectable)"],
        "diet": ["Complex carbohydrates with protein", "Small, frequent meals every 3-4 hours", "Nuts, whole grain toast with peanut butter", "Avoid isolated refined sugars"],
        "workout": ["Check blood sugar before exercising", "Carry fast-acting carbs during workout", "Avoid workouts during peak insulin activity"]
    },
    "Osteoarthritis": {
        "symptoms": ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"],
        "category": "Rheumatology & Orthopedics",
        "description": "Osteoarthritis is the most common form of arthritis, caused by the gradual breakdown of protective cartilage cushioning the ends of bones.",
        "precautions": ["Maintain a healthy weight to reduce joint load", "Apply hot or cold compresses for stiffness/inflammation", "Use supportive cushioned footwear or braces", "Stay physically active with low-impact routines"],
        "medications": ["Topical NSAIDs (Diclofenac gel)", "Oral analgesics (Acetaminophen, Celecoxib)", "Intra-articular hyaluronic acid / corticosteroid injections"],
        "diet": ["Anti-inflammatory Mediterranean diet", "Omega-3 rich fatty fish", "Berries, cherries, and olive oil", "Glucosamine / Chondroitin sources"],
        "workout": ["Swimming and water aerobics", "Stationary cycling", "Quadriceps and hamstring strengthening"]
    },
    "Arthritis": {
        "symptoms": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
        "category": "Rheumatology",
        "description": "Inflammatory arthritis involves swelling and tenderness of one or more joints, often autoimmune in origin (e.g., Rheumatoid Arthritis).",
        "precautions": ["Consult a rheumatologist early for disease-modifying therapy", "Protect joints from repetitive high-impact stress", "Use assistive devices when needed", "Get adequate restorative sleep"],
        "medications": ["DMARDs (Methotrexate, Leflunomide)", "Biologics (TNF inhibitors)", "NSAIDs for symptom relief"],
        "diet": ["Turmeric with black pepper (Curcumin)", "Ginger, walnuts, flaxseeds", "Leafy green vegetables", "Avoid ultra-processed foods"],
        "workout": ["Range-of-motion stretching", "Tai Chi and gentle yoga", "Warm-water hydrotherapy"]
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "symptoms": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
        "category": "Neurology & ENT",
        "description": "Benign Paroxysmal Positional Vertigo (BPPV) is a disorder caused by displaced calcium carbonate crystals in the inner ear canal.",
        "precautions": ["Perform canalith repositioning maneuvers (Epley maneuver) with a specialist", "Avoid sudden head movements or turning rapidly in bed", "Sleep with head slightly elevated on two pillows", "Sit down immediately when dizzy to prevent falls"],
        "medications": ["Vestibular suppressants for acute symptoms (Betahistine, Meclizine)", "Antiemetics"],
        "diet": ["Low sodium diet to maintain inner ear fluid balance", "Adequate hydration", "Avoid excess caffeine and alcohol"],
        "workout": ["Brandt-Daroff home exercises", "Balance and vestibular rehabilitation therapy"]
    },
    "Acne": {
        "symptoms": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
        "category": "Dermatology",
        "description": "Acne is an inflammatory skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
        "precautions": ["Wash face twice daily with a gentle, non-comedogenic cleanser", "Do not squeeze, pop, or pick at pimples to avoid scarring", "Use oil-free, non-comedogenic sunscreen and cosmetics", "Change pillowcases frequently"],
        "medications": ["Topical retinoids (Adapalene, Tretinoin)", "Benzoyl Peroxide", "Topical/Oral antibiotics (Clindamycin, Doxycycline)", "Isotretinoin for severe cystic acne"],
        "diet": ["Low-glycemic index foods", "Zinc and Vitamin A rich foods", "Green tea", "Reduce skim milk and whey protein intake"],
        "workout": ["Shower immediately after sweating", "Wear clean, loose-fitting athletic gear"]
    },
    "Urinary tract infection": {
        "symptoms": ["burning_micturition", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine"],
        "category": "Urology / Nephrology",
        "description": "A urinary tract infection (UTI) is an infection in any part of the urinary system, commonly the bladder and urethra.",
        "precautions": ["Drink plenty of water to flush bacteria from the urinary tract", "Wipe from front to back after urination", "Urinate promptly after intercourse", "Complete full prescribed antibiotic course"],
        "medications": ["Antibiotics (Nitrofurantoin, Fosfomycin, Trimethoprim-sulfamethoxazole)", "Urinary analgesic (Phenazopyridine)"],
        "diet": ["Unsweetened cranberry juice / D-mannose", "Plenty of clean water (2.5 - 3 liters daily)", "Probiotic yogurt", "Avoid bladder irritants like coffee, alcohol, and spicy foods"],
        "workout": ["Gentle walking; avoid intense pelvic strain or swimming until infection clears"]
    },
    "Psoriasis": {
        "symptoms": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
        "category": "Dermatology",
        "description": "Psoriasis is a chronic autoimmune skin condition that speeds up the life cycle of skin cells, leading to thick, red, scaly plaques.",
        "precautions": ["Apply rich moisturizers immediately after bathing", "Get moderate, safe sunlight exposure", "Avoid known skin injuries, stress, and smoking", "Consult a dermatologist for targeted systemic/biologic therapy"],
        "medications": ["Topical corticosteroids and Vitamin D analogues", "Calcineurin inhibitors", "Biologics (IL-17 / IL-23 inhibitors)", "Methotrexate"],
        "diet": ["Anti-inflammatory diet rich in Omega-3 fatty acids", "Turmeric, green leafy vegetables", "Gluten-free trial if sensitive", "Avoid alcohol and red meat"],
        "workout": ["Regular low-impact exercise like swimming (rinse chlorine afterwards) and walking", "Mindfulness stress-reduction"]
    },
    "Impetigo": {
        "symptoms": ["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
        "category": "Pediatric Dermatology / Infectious Disease",
        "description": "Impetigo is a highly contagious bacterial skin infection, most common in children, causing honey-colored crusting sores.",
        "precautions": ["Gently wash affected areas with antibacterial soap and warm water", "Cover sores loosely with gauze to prevent scratching and spread", "Wash hands frequently and do not share towels, sheets, or clothing", "Keep child home until 24-48 hours after starting antibiotics"],
        "medications": ["Topical Mupirocin or Retapamulin ointment", "Oral antibiotics (Cephalexin, Augmentin) for widespread lesions"],
        "diet": ["Nutrient-rich balanced diet", "Fresh fruits and protein for tissue healing", "Plenty of fluids"],
        "workout": ["Rest at home and avoid contact sports until lesions are healed"]
    }
}

# 132 Standard Symptoms
ALL_SYMPTOMS = sorted(list({
    symptom for data in DISEASES_DATA.values() for symptom in data["symptoms"]
} | {
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_ urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss",
    "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes",
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine",
    "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea",
    "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
    "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus",
    "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger",
    "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain",
    "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements",
    "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
    "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic _patches", "watering_from_eyes", "increased_appetite", "polyuria",
    "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", "blackheads",
    "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
    "blister", "red_sore_around_nose", "yellow_crust_ooze"
}))

# Symptom severity mapping (1 to 7)
SEVERITY_WEIGHTS = {
    "itching": 1, "skin_rash": 3, "nodal_skin_eruptions": 4, "continuous_sneezing": 4, "shivering": 5, "chills": 3,
    "joint_pain": 3, "stomach_pain": 5, "acidity": 3, "ulcers_on_tongue": 4, "muscle_wasting": 3, "vomiting": 5,
    "burning_micturition": 6, "spotting_ urination": 6, "fatigue": 4, "weight_gain": 3, "anxiety": 4,
    "cold_hands_and_feets": 5, "mood_swings": 3, "weight_loss": 3, "restlessness": 5, "lethargy": 2,
    "patches_in_throat": 6, "irregular_sugar_level": 5, "cough": 4, "high_fever": 7, "sunken_eyes": 3,
    "breathlessness": 6, "sweating": 3, "dehydration": 4, "indigestion": 5, "headache": 3, "yellowish_skin": 3,
    "dark_urine": 4, "nausea": 5, "loss_of_appetite": 4, "pain_behind_the_eyes": 4, "back_pain": 3,
    "constipation": 4, "abdominal_pain": 4, "diarrhoea": 6, "mild_fever": 5, "yellow_urine": 4,
    "yellowing_of_eyes": 4, "acute_liver_failure": 7, "fluid_overload": 6, "swelling_of_stomach": 7,
    "swelled_lymph_nodes": 6, "malaise": 6, "blurred_and_distorted_vision": 5, "phlegm": 5,
    "throat_irritation": 4, "redness_of_eyes": 5, "sinus_pressure": 4, "runny_nose": 5, "congestion": 5,
    "chest_pain": 7, "weakness_in_limbs": 7, "fast_heart_rate": 5, "pain_during_bowel_movements": 5,
    "pain_in_anal_region": 6, "bloody_stool": 5, "irritation_in_anus": 6, "neck_pain": 5, "dizziness": 4,
    "cramps": 4, "bruising": 4, "obesity": 4, "swollen_legs": 5, "swollen_blood_vessels": 5,
    "puffy_face_and_eyes": 5, "enlarged_thyroid": 6, "brittle_nails": 5, "swollen_extremeties": 5,
    "excessive_hunger": 4, "extra_marital_contacts": 5, "drying_and_tingling_lips": 4, "slurred_speech": 6,
    "knee_pain": 3, "hip_joint_pain": 2, "muscle_weakness": 2, "stiff_neck": 4, "swelling_joints": 5,
    "movement_stiffness": 5, "spinning_movements": 6, "loss_of_balance": 4, "unsteadiness": 4,
    "weakness_of_one_body_side": 7, "loss_of_smell": 3, "bladder_discomfort": 4, "foul_smell_of urine": 5,
    "continuous_feel_of_urine": 6, "passage_of_gases": 5, "internal_itching": 4, "toxic_look_(typhos)": 5,
    "depression": 3, "irritability": 2, "muscle_pain": 2, "altered_sensorium": 6, "red_spots_over_body": 3,
    "belly_pain": 4, "abnormal_menstruation": 6, "dischromic _patches": 6, "watering_from_eyes": 4,
    "increased_appetite": 5, "polyuria": 4, "family_history": 5, "mucoid_sputum": 4, "rusty_sputum": 4,
    "lack_of_concentration": 3, "visual_disturbances": 3, "receiving_blood_transfusion": 5,
    "receiving_unsterile_injections": 5, "coma": 7, "stomach_bleeding": 7, "distention_of_abdomen": 4,
    "history_of_alcohol_consumption": 5, "blood_in_sputum": 5, "prominent_veins_on_calf": 6,
    "palpitations": 4, "painful_walking": 2, "pus_filled_pimples": 2, "blackheads": 2, "scurring": 2,
    "skin_peeling": 3, "silver_like_dusting": 2, "small_dents_in_nails": 2, "inflammatory_nails": 2,
    "blister": 4, "red_sore_around_nose": 4, "yellow_crust_ooze": 4
}

def generate_symptom_dataset():
    """Generates the Kaggle-standard 42-disease x 132-symptom dataset with train/test samples."""
    np.random.seed(42)
    records = []
    
    for disease, details in DISEASES_DATA.items():
        base_symptoms = details["symptoms"]
        # Generate 120 synthetic variations per disease for a robust training distribution (~4,920 rows)
        for _ in range(120):
            row = {symptom: 0 for symptom in ALL_SYMPTOMS}
            
            # Select 70% to 100% of core symptoms
            k = np.random.randint(max(1, len(base_symptoms) - 1), len(base_symptoms) + 1)
            active_symptoms = np.random.choice(base_symptoms, size=k, replace=False)
            for s in active_symptoms:
                if s in row:
                    row[s] = 1
            
            # Add a small realistic noise symptom with 5% chance
            if np.random.rand() < 0.05:
                random_sym = np.random.choice(ALL_SYMPTOMS)
                row[random_sym] = 1
                
            row["prognosis"] = disease
            records.append(row)
            
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_RAW_DIR, "Training.csv"), index=False)
    
    # Also generate a clean Testing split
    test_records = []
    for disease, details in DISEASES_DATA.items():
        base_symptoms = details["symptoms"]
        for _ in range(20):
            row = {symptom: 0 for symptom in ALL_SYMPTOMS}
            for s in base_symptoms:
                if s in row:
                    row[s] = 1
            row["prognosis"] = disease
            test_records.append(row)
    df_test = pd.DataFrame(test_records)
    df_test.to_csv(os.path.join(DATA_RAW_DIR, "Testing.csv"), index=False)
    print(f"Generated Training.csv ({len(df)} rows) and Testing.csv ({len(df_test)} rows)")

def generate_disease_metadata_csvs():
    """Generates symptom description, precaution, medication, diet, and severity CSVs."""
    # 1. Symptom Severity
    sev_rows = [{"Symptom": sym, "weight": SEVERITY_WEIGHTS.get(sym, 3)} for sym in ALL_SYMPTOMS]
    pd.DataFrame(sev_rows).to_csv(os.path.join(DATA_RAW_DIR, "symptom_severity.csv"), index=False)

    # 2. Descriptions
    desc_rows = [{"Disease": dis, "Description": data["description"]} for dis, data in DISEASES_DATA.items()]
    pd.DataFrame(desc_rows).to_csv(os.path.join(DATA_RAW_DIR, "symptom_Description.csv"), index=False)

    # 3. Precautions
    prec_rows = []
    for dis, data in DISEASES_DATA.items():
        p = data["precautions"]
        prec_rows.append({
            "Disease": dis,
            "Precaution_1": p[0] if len(p) > 0 else "",
            "Precaution_2": p[1] if len(p) > 1 else "",
            "Precaution_3": p[2] if len(p) > 2 else "",
            "Precaution_4": p[3] if len(p) > 3 else ""
        })
    pd.DataFrame(prec_rows).to_csv(os.path.join(DATA_RAW_DIR, "symptom_precaution.csv"), index=False)

    # 4. Medications
    med_rows = []
    for dis, data in DISEASES_DATA.items():
        med_rows.append({
            "Disease": dis,
            "Medication": json.dumps(data["medications"])
        })
    pd.DataFrame(med_rows).to_csv(os.path.join(DATA_RAW_DIR, "disease_medications.csv"), index=False)

    # 5. Diet & Workout
    diet_rows = []
    for dis, data in DISEASES_DATA.items():
        diet_rows.append({
            "Disease": dis,
            "Diet": json.dumps(data["diet"]),
            "Workout": json.dumps(data["workout"]),
            "Category": data["category"]
        })
    pd.DataFrame(diet_rows).to_csv(os.path.join(DATA_RAW_DIR, "disease_lifestyle.csv"), index=False)

    print("Generated all disease metadata CSVs.")

def generate_diabetes_dataset():
    """Generates standard Pima Indians Diabetes Dataset format (~800 samples)."""
    np.random.seed(42)
    n = 800
    
    pregnancies = np.random.randint(0, 15, n)
    glucose = np.random.normal(120, 30, n).clip(50, 200).astype(int)
    bp = np.random.normal(70, 12, n).clip(40, 125).astype(int)
    skin_thickness = np.random.normal(20, 10, n).clip(5, 60).astype(int)
    insulin = np.random.normal(85, 60, n).clip(15, 600).astype(int)
    bmi = np.random.normal(32, 6.5, n).clip(18.0, 55.0).round(1)
    dpf = np.random.normal(0.47, 0.3, n).clip(0.08, 2.4).round(3)
    age = np.random.randint(21, 81, n)
    
    # Calculate probability score for diabetes
    score = (
        (glucose - 120) * 0.03 +
        (bmi - 30) * 0.08 +
        (age - 45) * 0.03 +
        (insulin - 100) * 0.004 +
        (dpf - 0.5) * 0.6 +
        (pregnancies - 3) * 0.05
    )
    prob = 1 / (1 + np.exp(-score))
    outcome = (prob > 0.50).astype(int)
    
    df = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Outcome": outcome
    })
    df.to_csv(os.path.join(DATA_RAW_DIR, "diabetes.csv"), index=False)
    print(f"Generated diabetes.csv ({len(df)} rows, {outcome.sum()} positive cases, {(1-outcome.mean())*100:.1f}% negative)")

def generate_heart_dataset():
    """Generates UCI Cleveland Heart Disease Dataset format (~700 samples)."""
    np.random.seed(42)
    n = 700
    
    age = np.random.randint(29, 78, n)
    sex = np.random.binomial(1, 0.68, n)
    cp = np.random.choice([0, 1, 2, 3], p=[0.47, 0.16, 0.28, 0.09], size=n)  # chest pain type
    trestbps = np.random.normal(131, 17, n).clip(94, 200).astype(int)  # resting BP
    chol = np.random.normal(246, 51, n).clip(126, 460).astype(int)  # serum cholestoral
    fbs = np.random.binomial(1, 0.15, n)  # fasting blood sugar > 120 mg/dl
    restecg = np.random.choice([0, 1, 2], p=[0.49, 0.48, 0.03], size=n)
    thalach = np.random.normal(149, 23, n).clip(71, 202).astype(int)  # max heart rate
    exang = np.random.binomial(1, 0.32, n)  # exercise induced angina
    oldpeak = np.random.exponential(1.0, n).clip(0.0, 6.2).round(1)  # ST depression
    slope = np.random.choice([0, 1, 2], p=[0.07, 0.46, 0.47], size=n)
    ca = np.random.choice([0, 1, 2, 3], p=[0.58, 0.22, 0.13, 0.07], size=n)
    thal = np.random.choice([1, 2, 3], p=[0.06, 0.54, 0.40], size=n)
    
    # Heart disease risk formula
    risk = (
        (age - 54) * 0.04 +
        (sex - 0.5) * 0.5 +
        (cp - 1.0) * 0.6 +
        (trestbps - 130) * 0.02 +
        (chol - 240) * 0.005 +
        (fbs - 0.2) * 0.4 +
        (150 - thalach) * 0.025 +
        (exang - 0.3) * 0.8 +
        (oldpeak - 1.0) * 0.5 +
        ca * 0.4 +
        (thal - 2) * 0.5
    )
    prob = 1 / (1 + np.exp(-risk))
    target = (prob > 0.50).astype(int)
    
    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target
    })
    df.to_csv(os.path.join(DATA_RAW_DIR, "heart.csv"), index=False)
    print(f"Generated heart.csv ({len(df)} rows, {target.sum()} positive cases, {(1-target.mean())*100:.1f}% negative)")

def generate_parkinsons_dataset():
    """Generates UCI Parkinson's Disease Biomedical Voice Measurement Dataset (~450 samples)."""
    np.random.seed(42)
    n = 450
    
    # Biomedical features
    fo = np.random.normal(154, 40, n).clip(88, 260)        # MDVP:Fo(Hz)
    fhi = (fo + np.random.uniform(10, 80, n)).clip(100, 350) # MDVP:Fhi(Hz)
    flo = (fo - np.random.uniform(5, 50, n)).clip(65, 240)   # MDVP:Flo(Hz)
    jitter_pct = np.random.uniform(0.001, 0.025, n).round(5)
    jitter_abs = np.random.uniform(0.00001, 0.0002, n).round(6)
    rap = np.random.uniform(0.0005, 0.015, n).round(5)
    ppq = np.random.uniform(0.0008, 0.018, n).round(5)
    ddp = (rap * 3).round(5)
    shimmer = np.random.uniform(0.009, 0.12, n).round(5)
    shimmer_db = (shimmer * 8.5).round(3)
    apq3 = np.random.uniform(0.004, 0.06, n).round(5)
    apq5 = np.random.uniform(0.005, 0.07, n).round(5)
    apq = np.random.uniform(0.007, 0.09, n).round(5)
    dda = (apq3 * 3).round(5)
    nhr = np.random.uniform(0.0006, 0.3, n).round(5)
    hnr = np.random.normal(21.8, 4.4, n).clip(8.4, 33.0).round(3)
    rpde = np.random.uniform(0.25, 0.68, n).round(4)
    dfa = np.random.uniform(0.57, 0.82, n).round(4)
    spread1 = np.random.uniform(-7.9, -2.4, n).round(3)
    spread2 = np.random.uniform(0.006, 0.45, n).round(4)
    d2 = np.random.uniform(1.4, 3.6, n).round(4)
    ppe = np.random.uniform(0.04, 0.52, n).round(4)
    
    # Parkinson's risk calculation from vocal instability features
    score = (
        (jitter_pct - 0.006) * 120 +
        (shimmer - 0.03) * 30 +
        (22 - hnr) * 0.2 +
        (rpde - 0.5) * 4.0 +
        (spread1 + 5.5) * 0.8 +
        (spread2 - 0.2) * 5.0 +
        (ppe - 0.2) * 6.0
    )
    prob = 1 / (1 + np.exp(-score))
    status = (prob > 0.50).astype(int)
    
    df = pd.DataFrame({
        "MDVP:Fo(Hz)": fo.round(2),
        "MDVP:Fhi(Hz)": fhi.round(2),
        "MDVP:Flo(Hz)": flo.round(2),
        "MDVP:Jitter(%)": jitter_pct,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe,
        "status": status
    })
    df.to_csv(os.path.join(DATA_RAW_DIR, "parkinsons.csv"), index=False)
    print(f"Generated parkinsons.csv ({len(df)} rows, {status.sum()} positive cases, {(1-status.mean())*100:.1f}% negative)")

if __name__ == "__main__":
    generate_symptom_dataset()
    generate_disease_metadata_csvs()
    generate_diabetes_dataset()
    generate_heart_dataset()
    generate_parkinsons_dataset()
    print("All datasets generated successfully!")
