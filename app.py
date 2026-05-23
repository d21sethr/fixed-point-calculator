import io
import base64
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

# FIX: Reverted to the standard Flask initialization. 
# Since your folder is correctly named "templates", Flask finds it automatically.
app = Flask(__name__)

def safe_parse_expression(expr_str):
    """
    Safely parses an input string into a SymPy expression.
    Replaces common notation like '^' with '**' and restricts available functions.
    """
    # Replace standard math notation if present
    expr_str = expr_str.replace('^', '**')
    
    # Define an explicit namespace for evaluation to prevent malicious input execution
    allowed_names = {
        'x': sp.Symbol('x'),
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'log': sp.log,
        'sqrt': sp.sqrt,
        'pi': sp.pi,
        'E': sp.E,
        'e': sp.E
    }
    
    try:
        # Evaluate without executing arbitrary raw string inputs
        parsed_expr = sp.parse_expr(expr_str, local_dict=allowed_names, evaluate=True)
        
        # Verify that only the variable 'x' is present
        free_symbols = parsed_expr.free_symbols
        if any(sym.name != 'x' for sym in free_symbols):
            raise ValueError("Only 'x' is permitted as an independent variable.")
        return parsed_expr
    except Exception as e:
        raise ValueError(f"Unable to parse expression: {str(e)}")

def evaluate_expression(expr, x_val):
    """Evaluates a SymPy expression at a specific numerical x value."""
    x = sp.Symbol('x')
    try:
        val = float(expr.subs(x, x_val).evalf())
        return val
    except Exception:
        raise ValueError(f"Evaluation failed at x = {x_val}.")

def generate_cobweb_plot(g_expr, iterations_data):
    """Generates a base64 encoded string of a Cobweb plot for the iterations."""
    if not iterations_data:
        return None
    
    x = sp.Symbol('x')
    x_vals = [step['x_n'] for step in iterations_data]
    
    # Establish graph domain limits
    min_x = min(x_vals)
    max_x = max(x_vals)
    padding = max(0.5, (max_x - min_x) * 0.3)
    x_start = min_x - padding
    x_end = max_x + padding
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    
    # Generate continuous values for curve plotting
    t_vals = np.linspace(x_start, x_end, 400)
    g_func = sp.lambdify(x, g_expr, modules=['numpy', {'sympy': lambda val: float(val.evalf())}])
    
    try:
        y_vals = []
        for val in t_vals:
            y_vals.append(float(g_expr.subs(x, val).evalf()))
        ax.plot(t_vals, y_vals, 'r-', label='$y = g(x)$', linewidth=2)
    except Exception:
        # Fallback if range mapping encounters numerical errors
        pass
        
    # Plot line y = x
    ax.plot(t_vals, t_vals, 'b--', label='$y = x$', linewidth=1.5)
    
    # Draw cobweb trajectory paths
    cobweb_x = []
    cobweb_y = []
    
    # Start trajectory line from (x0, 0)
    cobweb_x.extend([x_vals[0], x_vals[0]])
    cobweb_y.extend([0, x_vals[1] if len(x_vals) > 1 else x_vals[0]])
    
    for i in range(1, len(x_vals) - 1):
        curr_x = x_vals[i]
        next_x = x_vals[i+1]
        # Draw horizontal connector to y=x line
        cobweb_x.extend([curr_x, next_x])
        cobweb_y.extend([curr_x, curr_x])
        # Draw vertical connector to curve
        cobweb_x.extend([next_x, next_x])
        cobweb_y.extend([curr_x, next_x])
        
    ax.plot(cobweb_x, cobweb_y, 'g-', alpha=0.8, label='Iteration Path', linewidth=1.5)
    
    # Highlight solution point
    final_x = x_vals[-1]
    ax.plot(final_x, final_x, 'go', markersize=8, label=f'Root Approx: {final_x:.5f}')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Fixed-Point Iteration (Cobweb Plot)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')
    plt.tight_layout()
    
    # Output to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', results=None)

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        g_str = request.form.get('g_function', '').strip()
        x0_str = request.form.get('x0', '').strip()
        tol_str = request.form.get('tolerance', '').strip()
        max_iter_str = request.form.get('max_iterations', '').strip()
        
        # Validations
        if not g_str or not x0_str:
            return render_template('index.html', error="Please provide the equation $g(x)$ and an initial guess.")
        
        g_expr = safe_parse_expression(g_str)
        x0 = float(x0_str)
        tol = float(tol_str) if tol_str else 1e-5
        max_iter = int(max_iter_str) if max_iter_str else 50
        
        if max_iter <= 0 or max_iter > 250:
            return render_template('index.html', error="Maximum iterations must be an integer between 1 and 250.")
        
        if tol <= 0:
            return render_template('index.html', error="Tolerance must be a positive decimal float.")

        # Analytical Convergence Check at initial guess x0
        x = sp.Symbol('x')
        try:
            dg_dx = sp.diff(g_expr, x)
            dg_val = abs(float(dg_dx.subs(x, x0).evalf()))
            convergence_check = {
                'derivative_expr': str(dg_dx),
                'value_at_x0': round(dg_val, 5),
                'convergent': dg_val < 1.0
            }
        except Exception:
            convergence_check = {
                'derivative_expr': "Could not compute derivative analytically",
                'value_at_x0': "N/A",
                'convergent': None
            }

        # Iteration Execution
        iterations = []
        curr_x = x0
        converged = False
        
        # First entry: initial step
        iterations.append({
            'step': 0,
            'x_n': round(curr_x, 8),
            'g_xn': "N/A",
            'abs_error': "N/A"
        })
        
        for i in range(1, max_iter + 1):
            try:
                next_x = evaluate_expression(g_expr, curr_x)
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                return render_template('index.html', error=f"Mathematical calculation error encountered at Step {i}: {str(e)}")
            
            # Check for non-finite values (NaN/Inf)
            if not np.isfinite(next_x):
                return render_template('index.html', error=f"Calculated value became infinite or non-real at Step {i}.")

            abs_err = abs(next_x - curr_x)
            
            iterations.append({
                'step': i,
                'x_n': round(next_x, 8),
                'g_xn': round(next_x, 8), # since x_{n+1} = g(x_n)
                'abs_error': round(abs_err, 8)
            })
            
            if abs_err < tol:
                converged = True
                break
                
            curr_x = next_x
            
        # Build plot representation
        plot_data = generate_cobweb_plot(g_expr, iterations)
        
        results = {
            'g_str': g_str,
            'x0': x0,
            'tol': tol,
            'max_iter': max_iter,
            'converged': converged,
            'convergence_check': convergence_check,
            'iterations': iterations,
            'plot_data': plot_data
        }
        
        return render_template('index.html', results=results)
        
    except ValueError as val_err:
        return render_template('index.html', error=str(val_err))
    except Exception as e:
        return render_template('index.html', error=f"An unexpected process execution issue occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)