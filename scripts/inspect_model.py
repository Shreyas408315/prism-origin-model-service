import joblib
import json

def main():
    model_path = "model/prism_ensemble_clean.joblib"
    print(f"Loading artifact from: {model_path}")
    
    artifact = joblib.load(model_path)
    
    print("\n==================================================")
    print("MODEL ARTIFACT METADATA")
    print("==================================================")
    print(f"Type: {type(artifact)}")
    
    if isinstance(artifact, dict):
        print(f"Keys: {list(artifact.keys())}")
        
        print("\n--- Features ---")
        features = artifact.get("features", [])
        print(f"Count: {len(features)}")
        print(f"Names: {features}")
        
        print("\n--- Configuration ---")
        print(f"Threshold: {artifact.get('threshold')}")
        print(f"Positive Class: {artifact.get('positive_class')}")
        
        print("\n--- Components ---")
        models = artifact.get("models", {})
        print(f"Component count: {len(models)}")
        for name, comp in models.items():
            print(f"  - {name}: {type(comp)}")
            
        print("\n--- Weights ---")
        print(artifact.get("weights"))
    else:
        print("Artifact is not a dictionary.")

if __name__ == "__main__":
    main()
