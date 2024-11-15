from validation.experiment.runner import ValidationExperiment

def main():
    experiment = ValidationExperiment('templates/nanoGPT_lite/ideas/validation_experiment.json')
    results = experiment.run_experiment()
    print("\nExperiment Results:", results)

if __name__ == "__main__":
    main()
