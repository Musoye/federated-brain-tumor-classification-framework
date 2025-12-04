class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is not None:
            params, _ = aggregated

            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict)

            # Prune + Quantize before saving
            self.model = apply_pruning(self.model, amount=0.3)
            self.model = apply_quantization(self.model)

            save_path = f"{CONFIG['MODEL_NAME']}_final.pth"
            torch.save(self.model.state_dict(), save_path)

            print(f"✔ Saved final global model to {save_path}")

        return aggregated
