/**
 * Agentic Learning Platform - Main JavaScript
 * Interactive functionality for the agent design learning platform
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize all modules
    initNavigation();
    initScrollEffects();
    initPatternFilters();
    initExampleFilters();
    initPlayground();
    initLearningPaths();
    initCopyButtons();
    initSmoothScroll();
});

/**
 * Navigation Module
 * Handles mobile menu toggle and scroll effects
 */
function initNavigation() {
    const navbar = document.getElementById('navbar');
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');

    // Mobile menu toggle
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }

    // Close menu when clicking a link
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            navToggle.classList.remove('active');
        });
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
            navMenu.classList.remove('active');
            navToggle.classList.remove('active');
        }
    });

    // Scroll effect for navbar
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        lastScroll = currentScroll;
    });

    // Active link highlighting
    const sections = document.querySelectorAll('section[id]');

    function highlightNavLink() {
        const scrollY = window.pageYOffset;

        sections.forEach(section => {
            const sectionHeight = section.offsetHeight;
            const sectionTop = section.offsetTop - 100;
            const sectionId = section.getAttribute('id');

            if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    window.addEventListener('scroll', highlightNavLink);
}

/**
 * Scroll Effects Module
 * Intersection Observer for fade-in animations
 */
function initScrollEffects() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe cards and sections
    const animatedElements = document.querySelectorAll(
        '.pattern-card, .example-card, .overview-card, .resource-card, .learning-path'
    );

    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(el);
    });

    // Add visible class handler
    document.querySelectorAll('.pattern-card, .example-card, .overview-card, .resource-card, .learning-path').forEach(el => {
        el.addEventListener('transitionend', () => {
            if (el.classList.contains('visible')) {
                el.style.opacity = '';
                el.style.transform = '';
            }
        });
    });

    // Trigger visibility on intersection
    const makeVisible = new MutationObserver((mutations) => {
        mutations.forEach(mutation => {
            if (mutation.target.classList.contains('visible')) {
                mutation.target.style.opacity = '1';
                mutation.target.style.transform = 'translateY(0)';
            }
        });
    });

    animatedElements.forEach(el => {
        makeVisible.observe(el, { attributes: true, attributeFilter: ['class'] });
    });

    // Initial check for elements already in view
    animatedElements.forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.top < window.innerHeight) {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }
    });
}

/**
 * Pattern Filters Module
 * Filter pattern cards by category
 */
function initPatternFilters() {
    const filterBtns = document.querySelectorAll('.filter-bar .filter-btn');
    const patternCards = document.querySelectorAll('.pattern-card');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filter = btn.dataset.filter;

            // Filter cards
            patternCards.forEach(card => {
                const category = card.dataset.category;

                if (filter === 'all' || category === filter) {
                    card.style.display = 'block';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 10);
                } else {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            });
        });
    });
}

/**
 * Example Filters Module
 * Filter example cards by difficulty level
 */
function initExampleFilters() {
    const filterBtns = document.querySelectorAll('.examples-filter .example-filter-btn');
    const exampleCards = document.querySelectorAll('.example-card');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filter = btn.dataset.filter;

            // Filter cards
            exampleCards.forEach(card => {
                const level = card.dataset.level;

                if (filter === 'all' || level === filter) {
                    card.style.display = 'block';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 10);
                } else {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            });
        });
    });
}

/**
 * Playground Module
 * Interactive code generator based on configuration
 */
function initPlayground() {
    const agentType = document.getElementById('agent-type');
    const modelSelect = document.getElementById('model-select');
    const maxSteps = document.getElementById('max-steps');
    const stepsValue = document.getElementById('steps-value');
    const enablePlanning = document.getElementById('enable-planning');
    const taskPrompt = document.getElementById('task-prompt');

    // Tool checkboxes
    const toolSearch = document.getElementById('tool-search');
    const toolBrowse = document.getElementById('tool-browse');
    const toolImage = document.getElementById('tool-image');
    const toolCode = document.getElementById('tool-code');

    // Update steps display
    if (maxSteps && stepsValue) {
        maxSteps.addEventListener('input', () => {
            stepsValue.textContent = maxSteps.value;
            updatePlaygroundCode();
        });
    }

    // Add event listeners to all config elements
    const configElements = [agentType, modelSelect, enablePlanning, toolSearch, toolBrowse, toolImage, toolCode];
    configElements.forEach(el => {
        if (el) {
            el.addEventListener('change', updatePlaygroundCode);
        }
    });

    // Initial code generation
    updatePlaygroundCode();
}

/**
 * Generate playground code based on current configuration
 */
function updatePlaygroundCode() {
    const agentType = document.getElementById('agent-type')?.value || 'code';
    const modelSelect = document.getElementById('model-select')?.value || 'qwen-coder';
    const maxSteps = document.getElementById('max-steps')?.value || 10;
    const enablePlanning = document.getElementById('enable-planning')?.checked || false;
    const taskPrompt = document.getElementById('task-prompt')?.value || 'Your task here...';

    const toolSearch = document.getElementById('tool-search')?.checked || false;
    const toolBrowse = document.getElementById('tool-browse')?.checked || false;
    const toolImage = document.getElementById('tool-image')?.checked || false;
    const toolCode = document.getElementById('tool-code')?.checked || false;

    // Model mapping
    const modelMap = {
        'qwen-coder': 'Qwen/Qwen2.5-Coder-32B-Instruct',
        'llama': 'meta-llama/Llama-3.1-70B-Instruct',
        'gpt4': 'gpt-4o',
        'claude': 'claude-3-5-sonnet-20241022'
    };

    // Agent type mapping
    const agentClass = agentType === 'code' ? 'CodeAgent' : 'ToolCallingAgent';

    // Build imports
    let imports = [`from smolagents import ${agentClass}`];

    // Determine model import
    if (modelSelect === 'gpt4' || modelSelect === 'claude') {
        imports.push('from smolagents import LiteLLMModel');
    } else {
        imports.push('from smolagents import InferenceClientModel');
    }

    // Build tool imports
    const tools = [];
    const toolImports = [];

    if (toolSearch) {
        toolImports.push('DuckDuckGoSearchTool');
        tools.push('DuckDuckGoSearchTool()');
    }
    if (toolBrowse) {
        toolImports.push('VisitWebpageTool');
        tools.push('VisitWebpageTool()');
    }
    if (toolImage) {
        imports.push('from smolagents import load_tool');
    }
    if (toolCode) {
        toolImports.push('PythonInterpreterTool');
        tools.push('PythonInterpreterTool()');
    }

    if (toolImports.length > 0) {
        imports.push(`from smolagents import ${toolImports.join(', ')}`);
    }

    // Build tools list
    let toolsList = tools.length > 0 ? tools.join(', ') : '';

    // Add image tool loading if needed
    let imageToolSetup = '';
    if (toolImage) {
        imageToolSetup = `
# Load image generation tool from Hub
image_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
`;
        if (toolsList) {
            toolsList += ', image_tool';
        } else {
            toolsList = 'image_tool';
        }
    }

    // Build model initialization
    let modelInit;
    if (modelSelect === 'gpt4' || modelSelect === 'claude') {
        modelInit = `model = LiteLLMModel(model_id="${modelMap[modelSelect]}")`;
    } else {
        modelInit = `model = InferenceClientModel(\n    model_id="${modelMap[modelSelect]}"\n)`;
    }

    // Build agent initialization
    let agentConfig = [`    tools=[${toolsList}]`, '    model=model', `    max_steps=${maxSteps}`];

    if (enablePlanning) {
        agentConfig.push('    planning_interval=3  # Re-plan every 3 steps');
    }

    // Escape the task prompt for Python
    const escapedPrompt = taskPrompt.replace(/"/g, '\\"').replace(/\n/g, '\\n');

    // Generate final code
    const code = `${imports.join('\n')}
${imageToolSetup}
# Initialize the model
${modelInit}

# Create the agent
agent = ${agentClass}(
${agentConfig.join(',\n')}
)

# Run the agent
result = agent.run("${escapedPrompt}")
print(result)`;

    // Update the code display
    const codeElement = document.querySelector('#playground-code code');
    if (codeElement) {
        codeElement.textContent = code;
    }
}

/**
 * Copy playground code to clipboard
 */
function copyPlaygroundCode() {
    const codeElement = document.querySelector('#playground-code code');
    if (codeElement) {
        navigator.clipboard.writeText(codeElement.textContent).then(() => {
            showCopyFeedback(document.querySelector('.code-preview .copy-btn'));
        });
    }
}

/**
 * Learning Paths Module
 * Expandable/collapsible learning paths
 */
function initLearningPaths() {
    const pathHeaders = document.querySelectorAll('.path-header');

    pathHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const path = header.closest('.learning-path');
            const modules = path.querySelector('.path-modules');

            // Toggle expanded state
            if (modules.style.maxHeight) {
                modules.style.maxHeight = null;
                modules.style.padding = '0 32px';
            } else {
                modules.style.maxHeight = modules.scrollHeight + 32 + 'px';
                modules.style.padding = '0 32px 32px';
            }
        });
    });

    // Initialize all paths as expanded by default
    document.querySelectorAll('.path-modules').forEach(modules => {
        modules.style.maxHeight = modules.scrollHeight + 32 + 'px';
        modules.style.overflow = 'hidden';
        modules.style.transition = 'max-height 0.3s ease, padding 0.3s ease';
    });

    // Module click handler (for future progress tracking)
    document.querySelectorAll('.module').forEach(module => {
        module.style.cursor = 'pointer';
        module.addEventListener('click', () => {
            module.classList.toggle('completed');
            updatePathProgress(module.closest('.learning-path'));
        });
    });
}

/**
 * Update learning path progress
 */
function updatePathProgress(path) {
    const modules = path.querySelectorAll('.module');
    const completed = path.querySelectorAll('.module.completed');
    const progress = Math.round((completed.length / modules.length) * 100);

    const progressRing = path.querySelector('.progress-ring');
    if (progressRing) {
        progressRing.style.setProperty('--progress', progress);
        progressRing.querySelector('span').textContent = progress + '%';
    }
}

/**
 * Copy Code Buttons Module
 */
function initCopyButtons() {
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const codeBlock = this.closest('.pattern-code, .code-preview')?.querySelector('code');
            if (codeBlock) {
                navigator.clipboard.writeText(codeBlock.textContent).then(() => {
                    showCopyFeedback(this);
                });
            }
        });
    });
}

/**
 * Global copy function for onclick handlers
 */
function copyCode(btn) {
    const codeBlock = btn.closest('.pattern-code, .code-preview')?.querySelector('code');
    if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.textContent).then(() => {
            showCopyFeedback(btn);
        });
    }
}

/**
 * Show copy feedback on button
 */
function showCopyFeedback(btn) {
    const originalText = btn.textContent;
    btn.textContent = 'Copied!';
    btn.style.background = 'var(--secondary)';
    btn.style.borderColor = 'var(--secondary)';
    btn.style.color = 'white';

    setTimeout(() => {
        btn.textContent = originalText;
        btn.style.background = '';
        btn.style.borderColor = '';
        btn.style.color = '';
    }, 2000);
}

/**
 * Smooth Scroll Module
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offset = 80; // Navbar height
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Syntax Highlighting (Simple)
 * Basic syntax highlighting for Python code
 */
function highlightCode() {
    document.querySelectorAll('pre code').forEach(block => {
        let code = block.innerHTML;

        // Python keywords
        const keywords = ['from', 'import', 'class', 'def', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None', 'with', 'as', 'try', 'except', 'finally', 'raise', 'yield', 'lambda', 'pass', 'break', 'continue'];

        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b(${keyword})\\b`, 'g');
            code = code.replace(regex, '<span class="keyword">$1</span>');
        });

        // Strings
        code = code.replace(/(["'])((?:\\.|(?!\1)[^\\])*?)\1/g, '<span class="string">$1$2$1</span>');

        // Comments
        code = code.replace(/(#.*)$/gm, '<span class="comment">$1</span>');

        // Function calls
        code = code.replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g, '<span class="function">$1</span>(');

        block.innerHTML = code;
    });
}

// Make functions available globally
window.copyCode = copyCode;
window.copyPlaygroundCode = copyPlaygroundCode;
window.updatePlaygroundCode = updatePlaygroundCode;

// Run syntax highlighting after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Delay to ensure all content is loaded
    setTimeout(highlightCode, 100);
});
