from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # DONE: Implement for Task 1.1.
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)
    derivative_approximation = (f_plus - f_minus) / (2 * epsilon)
    return derivative_approximation


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # DONE: Implement for Task 1.4.
    visited = set()
    order: List[Variable] = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        if not v.is_leaf():
            for parent in v.parents:
                if not parent.is_constant():
                    dfs(parent)

        visited.add(v.unique_id)
        order.insert(0, v)

    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # DONE: Implement for Task 1.4.
    order = topological_sort(variable)
    derivs = {}
    derivs[variable.unique_id] = deriv
    for v in order:
        deriv = derivs[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(deriv)
        else:
            for var, der in v.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivs.setdefault(var.unique_id, 0.0)
                derivs[var.unique_id] = derivs[var.unique_id] + der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
