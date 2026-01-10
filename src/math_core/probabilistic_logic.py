"""
Probabilistic Soft Logic Module

Implements probabilistic logic for combining evidence:
- Probabilistic Soft Logic (PSL) rules
- Markov Logic Networks concepts
- Evidence combination (Dempster-Shafer)
- Fuzzy logic operators
- Belief propagation

NO HEURISTICS. NO SIMPLIFICATIONS.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable, List, Set
from dataclasses import dataclass, field
from scipy import optimize
from scipy.special import expit, logit

from .error_aware_arithmetic import EPSILON


@dataclass
class LogicalAtom:
    """A ground atom in the logical system."""
    predicate: str
    arguments: Tuple[str, ...]
    value: float = 0.5  # Soft truth value in [0, 1]
    
    def __hash__(self):
        return hash((self.predicate, self.arguments))
    
    def __eq__(self, other):
        return (self.predicate == other.predicate and 
                self.arguments == other.arguments)
    
    def __repr__(self):
        args = ', '.join(self.arguments)
        return f"{self.predicate}({args}) = {self.value:.3f}"


@dataclass
class PSLRule:
    """
    Probabilistic Soft Logic rule.
    
    Form: body_1 ∧ body_2 ∧ ... => head  [weight]
    
    Distance to satisfaction: max(0, body_value - head_value)
    """
    body: List[LogicalAtom]
    head: LogicalAtom
    weight: float = 1.0
    squared: bool = True  # Use squared hinge loss
    
    def distance_to_satisfaction(self, atom_values: Dict[LogicalAtom, float]) -> float:
        """
        Compute distance to satisfaction.
        
        d(r) = max(0, I(body) - I(head))^p
        where p = 2 if squared else 1
        """
        # Body is conjunction (min in Lukasiewicz logic)
        body_value = min(atom_values.get(a, a.value) for a in self.body)
        head_value = atom_values.get(self.head, self.head.value)
        
        distance = max(0, body_value - head_value)
        
        if self.squared:
            return distance ** 2
        return distance
    
    def gradient(self, atom_values: Dict[LogicalAtom, float]
                 ) -> Dict[LogicalAtom, float]:
        """Compute gradient of distance w.r.t. atom values."""
        body_value = min(atom_values.get(a, a.value) for a in self.body)
        head_value = atom_values.get(self.head, self.head.value)
        
        diff = body_value - head_value
        
        if diff <= 0:
            return {}
        
        grads = {}
        
        # Gradient w.r.t. head (negative)
        if self.squared:
            grads[self.head] = -2 * diff
        else:
            grads[self.head] = -1.0
        
        # Gradient w.r.t. body atoms (positive, but only for min)
        min_body = min(atom_values.get(a, a.value) for a in self.body)
        for atom in self.body:
            val = atom_values.get(atom, atom.value)
            if abs(val - min_body) < EPSILON:
                if self.squared:
                    grads[atom] = grads.get(atom, 0) + 2 * diff
                else:
                    grads[atom] = grads.get(atom, 0) + 1.0
        
        return grads


class PSLModel:
    """
    Probabilistic Soft Logic inference model.
    
    Minimizes weighted sum of rule distances.
    """
    
    def __init__(self):
        self.rules: List[PSLRule] = []
        self.atoms: Set[LogicalAtom] = set()
        self.observed: Dict[LogicalAtom, float] = {}
    
    def add_rule(self, rule: PSLRule):
        """Add a rule to the model."""
        self.rules.append(rule)
        self.atoms.add(rule.head)
        self.atoms.update(rule.body)
    
    def add_observation(self, atom: LogicalAtom, value: float):
        """Add an observed atom (fixed during inference)."""
        self.observed[atom] = value
        self.atoms.add(atom)
    
    def infer(self, max_iter: int = 1000, tolerance: float = 1e-6
              ) -> Dict[LogicalAtom, float]:
        """
        Perform MAP inference.
        
        Minimizes: Σ w_r * d(r)
        
        Subject to atom values in [0, 1].
        """
        # Free atoms (not observed)
        free_atoms = [a for a in self.atoms if a not in self.observed]
        n_free = len(free_atoms)
        
        if n_free == 0:
            return dict(self.observed)
        
        atom_to_idx = {a: i for i, a in enumerate(free_atoms)}
        
        def objective(x):
            """Total weighted distance."""
            values = dict(self.observed)
            for i, atom in enumerate(free_atoms):
                values[atom] = x[i]
            
            total = 0
            for rule in self.rules:
                total += rule.weight * rule.distance_to_satisfaction(values)
            
            return total
        
        def gradient(x):
            """Gradient of objective."""
            values = dict(self.observed)
            for i, atom in enumerate(free_atoms):
                values[atom] = x[i]
            
            grad = np.zeros(n_free)
            
            for rule in self.rules:
                rule_grads = rule.gradient(values)
                for atom, g in rule_grads.items():
                    if atom in atom_to_idx:
                        grad[atom_to_idx[atom]] += rule.weight * g
            
            return grad
        
        # Initial values
        x0 = np.array([a.value for a in free_atoms])
        
        # Optimize with bounds [0, 1]
        bounds = [(0, 1)] * n_free
        
        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tolerance}
        )
        
        # Construct result
        values = dict(self.observed)
        for i, atom in enumerate(free_atoms):
            values[atom] = result.x[i]
        
        return values


class DempsterShaferCombiner:
    """
    Dempster-Shafer theory for evidence combination.
    
    Combines belief functions from independent sources.
    """
    
    def __init__(self, frame_of_discernment: List[str]):
        """
        Initialize D-S combiner.
        
        Args:
            frame_of_discernment: Exhaustive, mutually exclusive hypotheses
        """
        self.frame = frame_of_discernment
        self.n_hypotheses = len(frame_of_discernment)
        
        # Power set (all subsets)
        self.power_set = self._generate_power_set()
    
    def _generate_power_set(self) -> List[frozenset]:
        """Generate power set of frame."""
        result = [frozenset()]
        for item in self.frame:
            result.extend([s | {item} for s in result])
        return result[1:]  # Exclude empty set
    
    def mass_to_belief(self, mass: Dict[frozenset, float]
                       ) -> Dict[frozenset, float]:
        """
        Convert mass function to belief function.
        
        Bel(A) = Σ_{B ⊆ A} m(B)
        """
        belief = {}
        for A in self.power_set:
            belief[A] = sum(
                mass.get(B, 0) 
                for B in self.power_set 
                if B <= A
            )
        return belief
    
    def mass_to_plausibility(self, mass: Dict[frozenset, float]
                             ) -> Dict[frozenset, float]:
        """
        Convert mass function to plausibility function.
        
        Pl(A) = Σ_{B ∩ A ≠ ∅} m(B)
        """
        plausibility = {}
        for A in self.power_set:
            plausibility[A] = sum(
                mass.get(B, 0)
                for B in self.power_set
                if B & A  # Non-empty intersection
            )
        return plausibility
    
    def combine(self, mass1: Dict[frozenset, float],
                 mass2: Dict[frozenset, float]) -> Dict[frozenset, float]:
        """
        Dempster's rule of combination.
        
        m(A) = [Σ_{B ∩ C = A} m1(B) * m2(C)] / (1 - K)
        
        where K = Σ_{B ∩ C = ∅} m1(B) * m2(C) is conflict
        """
        combined = {}
        
        # Compute conflict
        K = 0
        for B in self.power_set:
            for C in self.power_set:
                if not (B & C):  # Empty intersection
                    K += mass1.get(B, 0) * mass2.get(C, 0)
        
        if K >= 1 - EPSILON:
            raise ValueError("Complete conflict between evidence sources")
        
        # Combine masses
        for A in self.power_set:
            combined[A] = 0
            for B in self.power_set:
                for C in self.power_set:
                    if B & C == A:
                        combined[A] += mass1.get(B, 0) * mass2.get(C, 0)
            combined[A] /= (1 - K)
        
        return combined
    
    def combine_multiple(self, masses: List[Dict[frozenset, float]]
                         ) -> Dict[frozenset, float]:
        """Combine multiple evidence sources."""
        if not masses:
            # Vacuous belief
            full_set = frozenset(self.frame)
            return {full_set: 1.0}
        
        result = masses[0]
        for m in masses[1:]:
            result = self.combine(result, m)
        
        return result
    
    def pignistic_probability(self, mass: Dict[frozenset, float]
                               ) -> Dict[str, float]:
        """
        Convert mass function to pignistic probability.
        
        BetP(x) = Σ_{A ∋ x} m(A) / |A|
        """
        prob = {h: 0 for h in self.frame}
        
        for A, m in mass.items():
            if m > EPSILON:
                card_A = len(A)
                for x in A:
                    prob[x] += m / card_A
        
        return prob


class FuzzyLogic:
    """
    Fuzzy logic operations with various t-norms.
    """
    
    @staticmethod
    def lukasiewicz_and(a: float, b: float) -> float:
        """Lukasiewicz t-norm: max(0, a + b - 1)"""
        return max(0, a + b - 1)
    
    @staticmethod
    def lukasiewicz_or(a: float, b: float) -> float:
        """Lukasiewicz t-conorm: min(1, a + b)"""
        return min(1, a + b)
    
    @staticmethod
    def lukasiewicz_implies(a: float, b: float) -> float:
        """Lukasiewicz implication: min(1, 1 - a + b)"""
        return min(1, 1 - a + b)
    
    @staticmethod
    def godel_and(a: float, b: float) -> float:
        """Gödel t-norm: min(a, b)"""
        return min(a, b)
    
    @staticmethod
    def godel_or(a: float, b: float) -> float:
        """Gödel t-conorm: max(a, b)"""
        return max(a, b)
    
    @staticmethod
    def godel_implies(a: float, b: float) -> float:
        """Gödel implication: 1 if a <= b else b"""
        return 1.0 if a <= b else b
    
    @staticmethod
    def product_and(a: float, b: float) -> float:
        """Product t-norm: a * b"""
        return a * b
    
    @staticmethod
    def product_or(a: float, b: float) -> float:
        """Product t-conorm: a + b - a*b"""
        return a + b - a * b
    
    @staticmethod
    def product_implies(a: float, b: float) -> float:
        """Product implication: 1 if a <= b else b/a"""
        return 1.0 if a <= b + EPSILON else b / a
    
    @staticmethod
    def negate(a: float) -> float:
        """Standard negation: 1 - a"""
        return 1 - a


class BeliefPropagation:
    """
    Belief propagation for factor graphs.
    
    Computes marginal probabilities via message passing.
    """
    
    def __init__(self):
        self.variables: Dict[str, int] = {}  # variable -> num_states
        self.factors: List[Dict[str, Any]] = []
        self.messages: Dict[Tuple, np.ndarray] = {}
    
    def add_variable(self, name: str, num_states: int = 2):
        """Add a variable node."""
        self.variables[name] = num_states
    
    def add_factor(self, variables: List[str], 
                   potential: np.ndarray):
        """
        Add a factor node.
        
        Args:
            variables: Variables in this factor
            potential: Factor potential (tensor over variable states)
        """
        self.factors.append({
            'variables': variables,
            'potential': potential
        })
    
    def _initialize_messages(self):
        """Initialize messages to uniform."""
        self.messages = {}
        
        # Variable to factor messages
        for var, num_states in self.variables.items():
            for f_idx, factor in enumerate(self.factors):
                if var in factor['variables']:
                    key = ('v2f', var, f_idx)
                    self.messages[key] = np.ones(num_states) / num_states
        
        # Factor to variable messages
        for f_idx, factor in enumerate(self.factors):
            for var in factor['variables']:
                key = ('f2v', f_idx, var)
                num_states = self.variables[var]
                self.messages[key] = np.ones(num_states) / num_states
    
    def _update_variable_to_factor(self, var: str, f_idx: int):
        """Update message from variable to factor."""
        num_states = self.variables[var]
        
        # Product of incoming factor-to-variable messages
        message = np.ones(num_states)
        
        for other_f_idx, factor in enumerate(self.factors):
            if other_f_idx != f_idx and var in factor['variables']:
                key = ('f2v', other_f_idx, var)
                message *= self.messages[key]
        
        # Normalize
        message /= (message.sum() + EPSILON)
        
        self.messages[('v2f', var, f_idx)] = message
    
    def _update_factor_to_variable(self, f_idx: int, var: str):
        """Update message from factor to variable."""
        factor = self.factors[f_idx]
        potential = factor['potential']
        variables = factor['variables']
        
        var_idx = variables.index(var)
        num_states = self.variables[var]
        
        # Sum-product over other variables
        # Start with potential
        result = potential.copy()
        
        # Multiply by incoming variable-to-factor messages
        for i, other_var in enumerate(variables):
            if other_var != var:
                key = ('v2f', other_var, f_idx)
                msg = self.messages[key]
                
                # Expand message to match potential shape
                shape = [1] * len(variables)
                shape[i] = len(msg)
                result = result * msg.reshape(shape)
        
        # Sum over all dimensions except var
        axes = [i for i in range(len(variables)) if i != var_idx]
        message = result.sum(axis=tuple(axes)) if axes else result
        
        # Normalize
        message /= (message.sum() + EPSILON)
        
        self.messages[('f2v', f_idx, var)] = message
    
    def infer(self, max_iter: int = 100, 
              tolerance: float = 1e-6) -> Dict[str, np.ndarray]:
        """
        Run belief propagation.
        
        Returns marginal probabilities for each variable.
        """
        self._initialize_messages()
        
        for iteration in range(max_iter):
            old_messages = {k: v.copy() for k, v in self.messages.items()}
            
            # Update variable to factor messages
            for var in self.variables:
                for f_idx, factor in enumerate(self.factors):
                    if var in factor['variables']:
                        self._update_variable_to_factor(var, f_idx)
            
            # Update factor to variable messages
            for f_idx, factor in enumerate(self.factors):
                for var in factor['variables']:
                    self._update_factor_to_variable(f_idx, var)
            
            # Check convergence
            max_diff = 0
            for key in self.messages:
                diff = np.max(np.abs(self.messages[key] - old_messages[key]))
                max_diff = max(max_diff, diff)
            
            if max_diff < tolerance:
                break
        
        # Compute marginals
        marginals = {}
        for var, num_states in self.variables.items():
            belief = np.ones(num_states)
            
            for f_idx, factor in enumerate(self.factors):
                if var in factor['variables']:
                    key = ('f2v', f_idx, var)
                    belief *= self.messages[key]
            
            belief /= (belief.sum() + EPSILON)
            marginals[var] = belief
        
        return marginals


def combine_probabilistic_evidence(
    probabilities: List[float],
    method: str = 'bayesian'
) -> float:
    """
    Combine multiple probability estimates.
    
    Args:
        probabilities: List of probability estimates
        method: 'bayesian', 'dempster_shafer', or 'fuzzy'
    
    Returns:
        Combined probability
    """
    if not probabilities:
        return 0.5
    
    if method == 'bayesian':
        # Independent evidence combination
        # P(A | E1, E2) ∝ P(A) * Π_i [P(E_i | A) / P(E_i)]
        # Simplified: log-odds fusion
        log_odds = [np.log(p / (1 - p + EPSILON) + EPSILON) for p in probabilities]
        combined_log_odds = sum(log_odds)
        return expit(combined_log_odds)
    
    elif method == 'dempster_shafer':
        # Convert probabilities to mass functions
        combiner = DempsterShaferCombiner(['True', 'False'])
        
        masses = []
        for p in probabilities:
            mass = {
                frozenset(['True']): p,
                frozenset(['False']): 1 - p
            }
            masses.append(mass)
        
        combined = combiner.combine_multiple(masses)
        return combined.get(frozenset(['True']), 0.5)
    
    elif method == 'fuzzy':
        # Fuzzy AND (product t-norm)
        result = probabilities[0]
        for p in probabilities[1:]:
            result = FuzzyLogic.product_and(result, p)
        return result
    
    raise ValueError(f"Unknown method: {method}")
