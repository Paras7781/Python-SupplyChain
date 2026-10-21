"""In-process client for SupplyIQ model and assessment services."""
from __future__ import annotations

from ml.regional_risk.model_training import predict_regional_risk
from ml.inventory_supplier.model_training import predict_inventory_demand
from ml.fuel_weather.model_training import predict_fuel_consumption


class SupplyIQClient:
    """Run model and assessment operations directly in this process."""

    def predict_regional_risk(self, order_data: dict) -> dict:
        """Predict regional delivery risk with the local model."""
        return predict_regional_risk(order_data)

    def predict_inventory_demand(self, inv_data: dict) -> dict:
        """Predict inventory demand with the synthetic demonstration model."""
        return predict_inventory_demand(inv_data)

    def predict_fuel(self, fuel_data: dict, price: float | None = None) -> dict:
        """Predict fuel consumption with the local model."""
        return predict_fuel_consumption(
            fuel_data, fuel_price_per_liter=price
        )


# Singleton in-process client
_client: SupplyIQClient | None = None


def get_api_client() -> SupplyIQClient:
    global _client
    if _client is None:
        _client = SupplyIQClient()
    return _client
