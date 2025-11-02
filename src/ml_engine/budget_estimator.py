"""
Budget Estimation and Validation
================================

Estimates trip costs and validates budget constraints:
- Distributes budget across categories (hotels, food, transport, tickets)
- Estimates POI-specific costs based on category and price level
- Calculates daily and total trip costs
- Validates budget feasibility
- Provides cost breakdowns and warnings

Author: Hybrid Trip Planner Team
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import from Phase 1 & Phase 2
from ..data_pipeline.data_models import POI
from .route_optimizer import OptimizedRoute, RouteStop
from config import config


@dataclass
class CostBreakdown:
    """
    Cost breakdown for a trip component
    
    Attributes:
        hotels (float): Accommodation costs
        food (float): Food and dining costs
        transport (float): Transportation costs
        tickets (float): Entry fees and tickets
        total (float): Total cost
    """
    hotels: float
    food: float
    transport: float
    tickets: float
    
    @property
    def total(self) -> float:
        return self.hotels + self.food + self.transport + self.tickets


@dataclass
class DayBudget:
    """
    Budget estimate for one day
    
    Attributes:
        day_number (int): Day number
        costs (CostBreakdown): Cost breakdown
        poi_costs (List[Tuple[str, float]]): Individual POI costs
        is_within_budget (bool): Whether day is within allocated budget
        allocated_budget (float): Budget allocated for this day
    """
    day_number: int
    costs: CostBreakdown
    poi_costs: List[Tuple[str, float]]
    is_within_budget: bool
    allocated_budget: float


@dataclass
class TripBudget:
    """
    Complete trip budget estimate
    
    Attributes:
        total_budget (float): Total user budget
        estimated_costs (CostBreakdown): Estimated cost breakdown
        daily_budgets (List[DayBudget]): Per-day budgets
        is_feasible (bool): Whether trip fits in budget
        budget_variance (float): Difference from budget (negative = over)
        warnings (List[str]): Budget warnings
    """
    total_budget: float
    estimated_costs: CostBreakdown
    daily_budgets: List[DayBudget]
    is_feasible: bool
    budget_variance: float
    warnings: List[str]


class BudgetEstimator:
    """
    Budget estimation and validation engine
    """
    
    def __init__(self):
        """Initialize Budget Estimator with config distribution"""
        self.logger = logging.getLogger(__name__)
        
        # Budget distribution percentages from config
        self.budget_distribution = {
            'hotels': config.HOTEL_BUDGET_PERCENT / 100.0,      # 0.40
            'food': config.FOOD_BUDGET_PERCENT / 100.0,         # 0.25
            'transport': config.TRANSPORT_BUDGET_PERCENT / 100.0, # 0.15
            'tickets': config.TICKETS_BUDGET_PERCENT / 100.0    # 0.20
        }
        
        # Base cost estimates by category (in INR)
        self.base_costs = {
            'tourism': 200,      # Entry fees
            'historic': 150,
            'leisure': 100,
            'food': 300,         # Meal cost
            'shopping': 500,     # Shopping estimate
            'amenity': 100
        }
        
        # Price level multipliers
        self.price_multipliers = {
            1: 0.5,   # Cheap
            2: 1.0,   # Moderate
            3: 1.5,   # Expensive
            4: 2.5    # Very expensive
        }
        
        # Per-day base costs
        self.hotel_cost_per_night = 1500  # INR per night
        self.transport_cost_per_day = 300  # INR per day
        
        self.logger.info("Budget Estimator initialized")
    
    def estimate_trip_budget(self, routes: List[OptimizedRoute],
                            total_budget: float,
                            num_days: int,
                            currency: str = 'INR') -> TripBudget:
        """
        Estimate complete trip budget
        
        Args:
            routes (List[OptimizedRoute]): Optimized daily routes
            total_budget (float): User's total budget
            num_days (int): Number of trip days
            currency (str): Currency code
            
        Returns:
            TripBudget: Complete budget estimate
        """
        if not routes:
            self.logger.warning("No routes to estimate budget for")
            return self._create_empty_budget(total_budget)
        
        self.logger.info(f"Estimating budget for {num_days}-day trip (₹{total_budget})")
        
        # Calculate budget allocation
        allocated_budgets = self._allocate_budget(total_budget, num_days)
        
        # Estimate daily costs
        daily_budgets = []
        for route in routes:
            day_budget = self._estimate_day_budget(
                route, 
                allocated_budgets[route.day_number - 1]
            )
            daily_budgets.append(day_budget)
        
        # Calculate total estimated costs
        total_hotels = sum(db.costs.hotels for db in daily_budgets)
        total_food = sum(db.costs.food for db in daily_budgets)
        total_transport = sum(db.costs.transport for db in daily_budgets)
        total_tickets = sum(db.costs.tickets for db in daily_budgets)
        
        estimated_costs = CostBreakdown(
            hotels=total_hotels,
            food=total_food,
            transport=total_transport,
            tickets=total_tickets
        )
        
        # Check feasibility
        budget_variance = total_budget - estimated_costs.total
        is_feasible = budget_variance >= 0
        
        # Generate warnings
        warnings = self._generate_budget_warnings(
            total_budget, estimated_costs, daily_budgets
        )
        
        trip_budget = TripBudget(
            total_budget=total_budget,
            estimated_costs=estimated_costs,
            daily_budgets=daily_budgets,
            is_feasible=is_feasible,
            budget_variance=budget_variance,
            warnings=warnings
        )
        
        self.logger.info(
            f"Budget estimate: ₹{estimated_costs.total:.0f} / ₹{total_budget:.0f} "
            f"({'✓ feasible' if is_feasible else '✗ over budget'})"
        )
        
        return trip_budget
    
    def _allocate_budget(self, total_budget: float, num_days: int) -> List[float]:
        """
        Allocate total budget across days
        
        Args:
            total_budget (float): Total budget
            num_days (int): Number of days
            
        Returns:
            List[float]: Budget per day
        """
        # Simple equal distribution
        per_day = total_budget / num_days
        return [per_day] * num_days
    
    def _estimate_day_budget(self, route: OptimizedRoute,
                            allocated_budget: float) -> DayBudget:
        """
        Estimate budget for one day
        
        Args:
            route (OptimizedRoute): Day route
            allocated_budget (float): Budget allocated for day
            
        Returns:
            DayBudget: Day budget estimate
        """
        # Hotel cost (1 night)
        hotel_cost = self.hotel_cost_per_night
        
        # Transport cost (based on distance)
        transport_cost = self._estimate_transport_cost(route)
        
        # Ticket costs (from POIs)
        ticket_costs = []
        total_tickets = 0.0
        
        for stop in route.stops:
            cost = self._estimate_poi_cost(stop.poi)
            if cost > 0:
                ticket_costs.append((stop.poi.name, cost))
                total_tickets += cost
        
        # Food costs (estimate from POIs + baseline)
        food_cost = self._estimate_food_cost(route)
        
        costs = CostBreakdown(
            hotels=hotel_cost,
            food=food_cost,
            transport=transport_cost,
            tickets=total_tickets
        )
        
        is_within_budget = costs.total <= allocated_budget
        
        return DayBudget(
            day_number=route.day_number,
            costs=costs,
            poi_costs=ticket_costs,
            is_within_budget=is_within_budget,
            allocated_budget=allocated_budget
        )
    
    def _estimate_transport_cost(self, route: OptimizedRoute) -> float:
        """
        Estimate transportation cost for day
        
        Args:
            route (OptimizedRoute): Day route
            
        Returns:
            float: Transport cost estimate
        """
        base_cost = self.transport_cost_per_day
        
        # Add cost based on distance traveled
        if route.total_distance_km > 5:
            # Extra cost for longer distances
            extra_km = route.total_distance_km - 5
            extra_cost = extra_km * 20  # ₹20 per km over 5km
            return base_cost + extra_cost
        
        return base_cost
    
    def _estimate_poi_cost(self, poi: POI) -> float:
        """
        Estimate cost for visiting a POI
        
        Args:
            poi (POI): POI to estimate cost for
            
        Returns:
            float: Estimated cost
        """
        # Free POIs
        if poi.fee_required is False:
            return 0.0
        
        # Get base cost by category
        base_cost = self.base_costs.get(poi.category, 100)
        
        # Apply price level multiplier
        if poi.price_level is not None:
            multiplier = self.price_multipliers.get(poi.price_level, 1.0)
            return base_cost * multiplier
        
        # Fee required but no price info
        if poi.fee_required is True:
            return base_cost
        
        # Unknown - assume moderate
        return base_cost * 0.5
    
    def _estimate_food_cost(self, route: OptimizedRoute) -> float:
        """
        Estimate food cost for day
        
        Args:
            route (OptimizedRoute): Day route
            
        Returns:
            float: Food cost estimate
        """
        # Count food-related POIs
        food_pois = [
            stop for stop in route.stops 
            if stop.poi.category == 'food'
        ]
        
        # Base: 3 meals per day
        base_meals = 3
        
        # If food POIs included, reduce base meals
        meals_at_pois = len(food_pois)
        other_meals = max(0, base_meals - meals_at_pois)
        
        # Calculate cost
        poi_food_cost = sum(self._estimate_poi_cost(stop.poi) for stop in food_pois)
        other_meal_cost = other_meals * 200  # ₹200 per meal
        
        return poi_food_cost + other_meal_cost
    
    def _generate_budget_warnings(self, total_budget: float,
                                 estimated_costs: CostBreakdown,
                                 daily_budgets: List[DayBudget]) -> List[str]:
        """
        Generate budget warnings and recommendations
        
        Args:
            total_budget (float): Total budget
            estimated_costs (CostBreakdown): Estimated costs
            daily_budgets (List[DayBudget]): Daily budget details
            
        Returns:
            List[str]: Warning messages
        """
        warnings = []
        
        # Overall budget check
        if estimated_costs.total > total_budget:
            overage = estimated_costs.total - total_budget
            overage_pct = (overage / total_budget) * 100
            warnings.append(
                f"Trip is over budget by ₹{overage:.0f} ({overage_pct:.1f}%). "
                f"Consider reducing POIs or increasing budget."
            )
        
        # Check if any single category is too high
        if estimated_costs.hotels > total_budget * 0.5:
            warnings.append(
                "Hotel costs are very high. Consider budget accommodations."
            )
        
        if estimated_costs.food > total_budget * 0.35:
            warnings.append(
                "Food costs are high. Consider more budget-friendly dining options."
            )
        
        if estimated_costs.tickets > total_budget * 0.3:
            warnings.append(
                "Entry fees are high. Consider free or low-cost attractions."
            )
        
        # Check daily budgets
        over_budget_days = [db for db in daily_budgets if not db.is_within_budget]
        if over_budget_days:
            day_nums = [str(db.day_number) for db in over_budget_days]
            warnings.append(
                f"Days {', '.join(day_nums)} exceed allocated budget. "
                f"Consider rebalancing activities."
            )
        
        # Check if budget is very tight
        if estimated_costs.total > total_budget * 0.95:
            warnings.append(
                "Budget is very tight with minimal buffer. "
                "Consider adding 10-15% buffer for unexpected costs."
            )
        
        return warnings
    
    def _create_empty_budget(self, total_budget: float) -> TripBudget:
        """Create empty budget structure"""
        return TripBudget(
            total_budget=total_budget,
            estimated_costs=CostBreakdown(0, 0, 0, 0),
            daily_budgets=[],
            is_feasible=True,
            budget_variance=total_budget,
            warnings=["No routes to estimate budget for"]
        )
    
    def adjust_budget_for_category(self, trip_budget: TripBudget,
                                   category: str,
                                   new_amount: float) -> TripBudget:
        """
        Adjust budget allocation for a specific category
        
        Args:
            trip_budget (TripBudget): Current budget
            category (str): Category to adjust ('hotels', 'food', 'transport', 'tickets')
            new_amount (float): New amount for category
            
        Returns:
            TripBudget: Updated budget
        """
        # Create new cost breakdown
        costs_dict = {
            'hotels': trip_budget.estimated_costs.hotels,
            'food': trip_budget.estimated_costs.food,
            'transport': trip_budget.estimated_costs.transport,
            'tickets': trip_budget.estimated_costs.tickets
        }
        
        if category in costs_dict:
            costs_dict[category] = new_amount
        
        new_costs = CostBreakdown(**costs_dict)
        
        # Recalculate feasibility
        variance = trip_budget.total_budget - new_costs.total
        is_feasible = variance >= 0
        
        # Regenerate warnings
        warnings = self._generate_budget_warnings(
            trip_budget.total_budget,
            new_costs,
            trip_budget.daily_budgets
        )
        
        return TripBudget(
            total_budget=trip_budget.total_budget,
            estimated_costs=new_costs,
            daily_budgets=trip_budget.daily_budgets,
            is_feasible=is_feasible,
            budget_variance=variance,
            warnings=warnings
        )
    
    def get_budget_statistics(self, trip_budget: TripBudget) -> Dict:
        """
        Get budget statistics for analysis
        
        Args:
            trip_budget (TripBudget): Trip budget
            
        Returns:
            Dict: Budget statistics
        """
        costs = trip_budget.estimated_costs
        
        return {
            'total_budget': trip_budget.total_budget,
            'total_estimated': costs.total,
            'budget_utilization_pct': (costs.total / trip_budget.total_budget) * 100,
            'remaining_budget': trip_budget.budget_variance,
            'cost_breakdown_pct': {
                'hotels': (costs.hotels / costs.total) * 100 if costs.total > 0 else 0,
                'food': (costs.food / costs.total) * 100 if costs.total > 0 else 0,
                'transport': (costs.transport / costs.total) * 100 if costs.total > 0 else 0,
                'tickets': (costs.tickets / costs.total) * 100 if costs.total > 0 else 0
            },
            'cost_breakdown_amount': {
                'hotels': costs.hotels,
                'food': costs.food,
                'transport': costs.transport,
                'tickets': costs.tickets
            },
            'daily_costs': [
                {
                    'day': db.day_number,
                    'total': db.costs.total,
                    'allocated': db.allocated_budget,
                    'within_budget': db.is_within_budget
                }
                for db in trip_budget.daily_budgets
            ],
            'is_feasible': trip_budget.is_feasible,
            'warnings_count': len(trip_budget.warnings)
        }
    
    def optimize_budget_allocation(self, trip_budget: TripBudget,
                                   target_category: str,
                                   max_adjustment_pct: float = 10) -> TripBudget:
        """
        Optimize budget allocation to reduce costs in target category
        
        Args:
            trip_budget (TripBudget): Current budget
            target_category (str): Category to optimize
            max_adjustment_pct (float): Max adjustment percentage
            
        Returns:
            TripBudget: Optimized budget
        """
        costs = trip_budget.estimated_costs
        costs_dict = {
            'hotels': costs.hotels,
            'food': costs.food,
            'transport': costs.transport,
            'tickets': costs.tickets
        }
        
        if target_category not in costs_dict:
            return trip_budget
        
        # Reduce target category by max_adjustment_pct
        current = costs_dict[target_category]
        reduction = current * (max_adjustment_pct / 100)
        costs_dict[target_category] = current - reduction
        
        new_costs = CostBreakdown(**costs_dict)
        variance = trip_budget.total_budget - new_costs.total
        is_feasible = variance >= 0
        
        warnings = self._generate_budget_warnings(
            trip_budget.total_budget,
            new_costs,
            trip_budget.daily_budgets
        )
        
        return TripBudget(
            total_budget=trip_budget.total_budget,
            estimated_costs=new_costs,
            daily_budgets=trip_budget.daily_budgets,
            is_feasible=is_feasible,
            budget_variance=variance,
            warnings=warnings
        )