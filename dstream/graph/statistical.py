from scipy import stats

class StatisticalAnnotator:
    
    @staticmethod
    def correlation_text(ax, x, y, data):
        
        r, p = stats.pearsonr(data[x], data[y])
        text = f"r = {r:.3f}, p = {p:.3e}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
         
    @staticmethod
    def regression_line(ax, x, y, data, color="red"):
         
        slope, intercept, r, p, _ = stats.linregress(data[x], data[y])
        ax.plot(data[x], slope * data[x] + intercept, color=color, lw=2, label=f"Fit (r={r:.2f})")
        ax.legend()
         


