def generate_basis_vectors(cavs, n_components=5, device='cuda:0'):
    """
    Find basis vectors that can represent multiple CAVs.
    
    Args:
        cavs (dict): Dictionary mapping style names to CAVs
        n_components (int): Number of components to find
        device (str): Device to use
        
    Returns:
        dict: Basis vectors and analysis results
    """
    from sklearn.decomposition import PCA
    
    # Stack and reshape CAVs for analysis
    cav_tensors = []
    style_names = []
    
    for style_name, cav in cavs.items():
        flat_cav = cav.cpu().view(-1).numpy()
        cav_tensors.append(flat_cav)
        style_names.append(style_name)
    
    cav_matrix = np.stack(cav_tensors)
    
    # Perform PCA to find principal components
    pca = PCA(n_components=min(n_components, len(cav_tensors)))
    pca_result = pca.fit_transform(cav_matrix)
    
    # Reshape components back to CAV shape
    original_shape = next(iter(cavs.values())).shape
    basis_vectors = []
    
    for component in pca.components_:
        basis_vector = torch.tensor(
            component.reshape(original_shape), 
            device=device
        )
        basis_vectors.append(basis_vector)
    
    # Calculate projection coefficients for each style
    projection_coefficients = {}
    for i, style_name in enumerate(style_names):
        # Get coefficients from PCA result
        coeffs = pca_result[i]
        projection_coefficients[style_name] = coeffs
    
    return {
        'basis_vectors': basis_vectors,
        'explained_variance': pca.explained_variance_ratio_,
        'projection_coefficients': projection_coefficients,
        'style_names': style_names
    }


def learn_multiple_cavs(cav_learner, content_img, style_classes, reference_class=None, batch_size=4):
    """
    Learn multiple CAVs for different style classes.
    
    Args:
        cav_learner (CAVLearner): CAV learner object
        content_img (torch.Tensor): Content image tensor
        style_classes (dict): Dictionary mapping class names to lists of style images
        reference_class (str, optional): Name of reference class for comparison
        batch_size (int): Batch size for processing
        
    Returns:
        dict: Dictionary mapping class names to CAVs
    """
    cavs = {}
    
    # If reference class is provided, use it as negative for all other classes
    if reference_class is not None and reference_class in style_classes:
        reference_styles = style_classes[reference_class]
        
        for class_name, styles in style_classes.items():
            if class_name != reference_class:
                logging.info(f"Learning CAV for {class_name} (vs {reference_class})")
                cav = cav_learner.learn_cav(
                    content_img,
                    styles,              # positive class
                    reference_styles,    # negative class
                    batch_size=batch_size
                )
                cavs[class_name] = cav
    # Otherwise, learn CAVs between each class and all others
    else:
        for class_name, styles in style_classes.items():
            # Collect all other styles as negative class
            negative_styles = []
            for other_class, other_styles in style_classes.items():
                if other_class != class_name:
                    negative_styles.extend(other_styles)
            
            if negative_styles:
                logging.info(f"Learning CAV for {class_name} (vs all others)")
                cav = cav_learner.learn_cav(
                    content_img,
                    styles,           # positive class
                    negative_styles,  # negative class
                    batch_size=batch_size
                )
                cavs[class_name] = cav
    
    return cavs


def visualize_basis_components(basis_result, output_dir):
    """
    Visualize PCA basis components of multiple CAVs.
    
    Args:
        basis_result (dict): Result from generate_basis_vectors
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot explained variance
    plt.figure(figsize=(10, 5))
    explained_variance = basis_result['explained_variance']
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative', color='red')
    
    plt.title('Explained Variance by Basis Components')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'explained_variance.png'))
    plt.close()
    
    # Visualize basis vectors
    n_basis = len(basis_result['basis_vectors'])
    plt.figure(figsize=(4 * min(n_basis, 5), 4 * ((n_basis + 4) // 5)))
    
    for idx, basis_vector in enumerate(basis_result['basis_vectors']):
        plt.subplot(((n_basis + 4) // 5), min(n_basis, 5), idx + 1)
        spatial_importance = torch.norm(basis_vector, dim=0).squeeze().cpu()
        plt.imshow(spatial_importance, cmap='viridis')
        plt.title(f'Basis {idx+1}\nVar: {basis_result["explained_variance"][idx]:.3f}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'basis_vectors.png'))
    plt.close()
    
    # Visualize style projections
    projection_coeffs = basis_result['projection_coefficients']
    style_names = basis_result['style_names']
    
    # Prepare data for visualization
    if len(style_names) > 1:
        # For 2+ components, make scatter plots
        if n_basis >= 2:
            plt.figure(figsize=(8, 8))
            
            # Extract first two components
            x = [projection_coeffs[style][0] for style in style_names]
            y = [projection_coeffs[style][1] for style in style_names]
            
            plt.scatter(x, y)
            
            # Add style names as labels
            for i, style in enumerate(style_names):
                plt.annotate(style, (x[i], y[i]))
            
            plt.title('Style Projection onto First Two Principal Components')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'style_projection_2d.png'))
            plt.close()
        
        # If we have 3+ components, make a 3D scatter plot
        if n_basis >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract first three components
            x = [projection_coeffs[style][0] for style in style_names]
            y = [projection_coeffs[style][1] for style in style_names]
            z = [projection_coeffs[style][2] for style in style_names]
            
            ax.scatter(x, y, z)
            
            # Add style names as labels
            for i, style in enumerate(style_names):
                ax.text(x[i], y[i], z[i], style)
            
            ax.set_title('Style Projection onto First Three Principal Components')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.savefig(os.path.join(output_dir, 'style_projection_3d.png'))
            plt.close()
    
    # Save projection coefficients
    with open(os.path.join(output_dir, 'projection_coefficients.txt'), 'w') as f:
        f.write("Style Projection Coefficients:\n\n")
        for style in style_names:
            f.write(f"{style}:\n")
            for i, coeff in enumerate(projection_coeffs[style]):
                f.write(f"  Component {i+1}: {coeff:.4f}\n")
            f.write("\n")


def style_interpolation_experiment(cav_transfer, cav_basis, content_img, style_img, output_dir):
    """
    Experiment with style interpolation using basis vectors.
    
    Args:
        cav_transfer (CAVStyleTransfer): CAV style transfer model
        cav_basis (dict): Result from generate_basis_vectors
        content_img (torch.Tensor): Content image tensor
        style_img (torch.Tensor): Style image tensor
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    basis_vectors = cav_basis['basis_vectors']
    n_basis = len(basis_vectors)
    
    if n_basis < 2:
        logging.warning("Not enough basis vectors for interpolation")
        return
    
    # Create grid of interpolation coefficients
    if n_basis == 2:
        # 2D grid for 2 components
        grid_size = 5
        coeff1_range = np.linspace(-2.0, 2.0, grid_size)
        coeff2_range = np.linspace(-2.0, 2.0, grid_size)
        
        plt.figure(figsize=(12, 12))
        
        for i, coeff1 in enumerate(coeff1_range):
            for j, coeff2 in enumerate(coeff2_range):
                # Create coefficient vector
                coefficients = [coeff1, coeff2] + [0.0] * (n_basis - 2)
                
                # Apply CAV combination
                combined_cav = torch.zeros_like(basis_vectors[0])
                for basis_vector, coeff in zip(basis_vectors, coefficients):
                    combined_cav += coeff * basis_vector
                
                # Apply to style transfer
                transfer = cav_transfer.apply_cav(
                    content_img, style_img, combined_cav, strength=1.0
                )
                
                # Plot in grid
                plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
                plt.imshow(transfer.squeeze(0).cpu().permute(1, 2, 0))
                plt.title(f"({coeff1:.1f}, {coeff2:.1f})")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'interpolation_grid.png'))
        plt.close()
    
    elif n_basis >= 3:
        # Create specific combinations for more than 2 components
        combinations = [
            (2.0, 0.0, 0.0),  # Pure component 1
            (0.0, 2.0, 0.0),  # Pure component 2
            (0.0, 0.0, 2.0),  # Pure component 3
            (1.0, 1.0, 0.0),  # Mix 1 & 2
            (1.0, 0.0, 1.0),  # Mix 1 & 3
            (0.0, 1.0, 1.0),  # Mix 2 & 3
            (1.0, 1.0, 1.0),  # Mix all
            (-1.0, -1.0, -1.0),  # Negative mix
            (2.0, -1.0, 0.0),  # Complex mix 1
            (0.0, 2.0, -1.0),  # Complex mix 2
        ]
        
        plt.figure(figsize=(15, 12))
        
        for i, coeffs in enumerate(combinations):
            # Pad coefficients if needed
            coefficients = list(coeffs) + [0.0] * (n_basis - len(coeffs))
            
            # Apply CAV combination
            combined_cav = torch.zeros_like(basis_vectors[0])
            for basis_vector, coeff in zip(basis_vectors, coefficients):
                combined_cav += coeff * basis_vector
            
            # Apply to style transfer
            transfer = cav_transfer.apply_cav(
                content_img, style_img, combined_cav, strength=1.0
            )
            
            # Plot
            plt.subplot(3, 4, i + 1)
            plt.imshow(transfer.squeeze(0).cpu().permute(1, 2, 0))
            plt.title(f"Coeffs: {coeffs}")
            plt.axis('off')
        
        # Also show original content and style
        plt.subplot(3, 4, 11)
        plt.imshow(content_img.squeeze(0).cpu().permute(1, 2, 0))
        plt.title("Content")
        plt.axis('off')
        
        plt.subplot(3, 4, 12)
        plt.imshow(style_img.squeeze(0).cpu().permute(1, 2, 0))
        plt.title("Style")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'basis_combinations.png'))
        plt.close()


def multi_style_experiment():
    """
    Extended experiment with multiple style classes and basis vectors.
    This function can be called instead of main() for more advanced experiments.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-Style CAV Experiment")
    
    # Directories
    parser.add_argument("--content_dir", type=str, default="./examples/content",
                      help="Directory containing content images")
    parser.add_argument("--content_segment_dir", type=str, default="./examples/content_segment",
                      help="Directory containing content segmentation masks")
    parser.add_argument("--output_dir", type=str, default="./multi_style_outputs",
                      help="Directory to save outputs")
    
    # Style class directories
    parser.add_argument("--ukiyo_e_dir", type=str, default="./styles/ukiyo_e",
                      help="Directory containing Ukiyo-e style images")
    parser.add_argument("--impressionist_dir", type=str, default="./styles/impressionist",
                      help="Directory containing Impressionist style images")
    parser.add_argument("--cubism_dir", type=str, default="./styles/cubism",
                      help="Directory containing Cubism style images")
    parser.add_argument("--abstract_dir", type=str, default="./styles/abstract",
                      help="Directory containing Abstract style images")
    parser.add_argument("--realism_dir", type=str, default="./styles/realism",
                      help="Directory containing Realism style images")
    
    # Segment directories
    parser.add_argument("--ukiyo_e_segment_dir", type=str, default=None,
                      help="Directory containing Ukiyo-e style segmentation masks")
    parser.add_argument("--impressionist_segment_dir", type=str, default=None,
                      help="Directory containing Impressionist style segmentation masks")
    parser.add_argument("--cubism_segment_dir", type=str, default=None,
                      help="Directory containing Cubism style segmentation masks")
    parser.add_argument("--abstract_segment_dir", type=str, default=None,
                      help="Directory containing Abstract style segmentation masks")
    parser.add_argument("--realism_segment_dir", type=str, default=None,
                      help="Directory containing Realism style segmentation masks")
    
    # Model parameters
    parser.add_argument("--wct_model_path", type=str, default="./model_checkpoints",
                      help="Path to WCT model checkpoints")
    parser.add_argument("--layer_index", type=int, default=3,
                      help="Layer index for feature extraction (1-4)")
    parser.add_argument("--option_unpool", type=str, default="cat5", choices=["sum", "cat5"],
                      help="Unpooling method")
    
    # Experiment parameters
    parser.add_argument("--image_size", type=int, default=256,
                      help="Size to resize images to")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for processing")
    parser.add_argument("--max_images", type=int, default=20,
                      help="Maximum number of images to load from each directory")
    parser.add_argument("--alpha", type=float, default=1.0,
                      help="Style transfer strength (0-1)")
    parser.add_argument("--svm_c", type=float, default=0.1,
                      help="SVM regularization parameter")
    parser.add_argument("--cav_strengths", type=float, nargs="+",
                      default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
                      help="CAV strengths to apply")
    parser.add_argument("--n_components", type=int, default=5,
                      help="Number of basis components to generate")
    parser.add_argument("--reference_class", type=str, default="Realism",
                      help="Reference style class for CAV learning")
    
    # System parameters
    parser.add_argument("--cpu", action="store_true",
                      help="Use CPU instead of GPU")
    parser.add_argument("--verbose", action="store_true",
                      help="Print detailed information")
    
    args = parser.parse_args()
    
    # Set device
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0'
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    comparison_dir = os.path.join(args.output_dir, 'comparisons')
    individual_dir = os.path.join(args.output_dir, 'individual')
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    basis_dir = os.path.join(args.output_dir, 'basis')
    interpolation_dir = os.path.join(args.output_dir, 'interpolation')
    
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(basis_dir, exist_ok=True)
    os.makedirs(interpolation_dir, exist_ok=True)
    
    # Initialize models
    logging.info("Initializing models")
    cav_transfer = CAVStyleTransfer(
        wct_model_path=args.wct_model_path,
        option_unpool=args.option_unpool,
        layer_index=args.layer_index,
        device=device,
        verbose=args.verbose
    )
    
    cav_learner = CAVLearner(cav_transfer, device=device)
    visualizer = StyleVisualizer(len(args.cav_strengths))
    cav_analyzer = CAVAnalyzer(device=device)
    
    # Load content images
    content_imgs = load_images(args.content_dir, args.image_size, args.max_images)
    content_segments = load_segments(args.content_segment_dir, args.image_size, args.max_images)
    
    # Move content to device
    content_imgs = [img.to(device) for img in content_imgs]
    
    # Define style class directories and their corresponding segmentation directories
    style_class_dirs = {
        "Ukiyo-e": args.ukiyo_e_dir,
        "Impressionist": args.impressionist_dir,
        "Cubism": args.cubism_dir,
        "Abstract": args.abstract_dir,
        "Realism": args.realism_dir
    }
    
    style_segment_dirs = {
        "Ukiyo-e": args.ukiyo_e_segment_dir,
        "Impressionist": args.impressionist_segment_dir,
        "Cubism": args.cubism_segment_dir,
        "Abstract": args.abstract_segment_dir,
        "Realism": args.realism_segment_dir
    }
    
    # Load style class images and segments
    style_classes = {}
    style_segments = {}
    
    for class_name, class_dir in style_class_dirs.items():
        if os.path.exists(class_dir):
            # Load style images
            logging.info(f"Loading style class: {class_name}")
            style_classes[class_name] = load_images(class_dir, args.image_size, args.max_images)
            
            # Move to device
            style_classes[class_name] = [style.to(device) for style in style_classes[class_name]]
            
            # Load segmentation masks if available
            segment_dir = style_segment_dirs[class_name]
            if segment_dir and os.path.exists(segment_dir):
                style_segments[class_name] = load_segments(segment_dir, args.image_size, args.max_images)
    
    # Check if we have enough style classes
    if len(style_classes) < 2:
        logging.error("At least two style classes are required for multi-style experiment")
        return
    
    # Ensure reference class exists
    if args.reference_class not in style_classes:
        logging.warning(f"Reference class {args.reference_class} not found, using first available class")
        reference_class = next(iter(style_classes.keys()))
    else:
        reference_class = args.reference_class
    
    # Learn CAVs for each style class
    logging.info("Learning CAVs for each style class")
    cavs = {}
    
    for class_name, styles in style_classes.items():
        if class_name != reference_class:
            logging.info(f"Learning CAV for {class_name} (vs {reference_class})")
            
            # Get segments if available
            class_segments = style_segments.get(class_name)
            reference_segments = style_segments.get(reference_class)
            
            # Learn CAV
            cav = cav_learner.learn_cav(
                content_imgs[0],           # Use first content image
                styles,                    # Positive class
                style_classes[reference_class],  # Negative class (reference)
                class_segments,            # Positive segments
                reference_segments,        # Negative segments
                batch_size=args.batch_size,
                c_param=args.svm_c
            )
            
            cavs[class_name] = cav
            
            # Save CAV
            torch.save(cav, os.path.join(args.output_dir, f'cav_{class_name}_vs_{reference_class}.pt'))
            
            # Analyze CAV
            cav_analyzer.visualize_cav(cav, os.path.join(analysis_dir, class_name))
            
            # Apply CAV to first content image and visualize results
            style_img = styles[0]  # Use first style image
            transfers = []
            
            for strength in args.cav_strengths:
                # Apply CAV with specified strength
                transfer = cav_transfer.apply_cav(
                    content_imgs[0], 
                    style_img, 
                    cav, 
                    strength,
                    alpha=args.alpha
                )
                
                transfers.append(transfer)
                
                # Save individual result
                output_filename = f'{class_name}_strength_{strength:.1f}.png'
                output_path = os.path.join(individual_dir, output_filename)
                save_image(transfer, output_path)
            
            # Create comparison plot
            visualizer.create_comparison_plot(
                transfers,
                args.cav_strengths,
                content_imgs[0],
                style_img,
                comparison_dir,
                'content_0',
                class_name
            )
    
    # If we have multiple CAVs, generate and analyze basis vectors
    if len(cavs) > 1:
        logging.info("Generating basis vectors for multiple CAVs")
        
        # Generate basis vectors
        basis_result = generate_basis_vectors(
            cavs,
            n_components=min(args.n_components, len(cavs)),
            device=device
        )
        
        # Save basis vectors
        torch.save(basis_result, os.path.join(args.output_dir, 'cav_basis_vectors.pt'))
        
        # Visualize basis components
        visualize_basis_components(basis_result, basis_dir)
        
        # Run interpolation experiments
        logging.info("Running style interpolation experiments")
        
        # For each style class, run interpolation experiment
        for class_name, styles in style_classes.items():
            # Skip reference class as base style
            if class_name == reference_class:
                continue
            
            # Run interpolation experiment
            output_subdir = os.path.join(interpolation_dir, f'base_{class_name}')
            os.makedirs(output_subdir, exist_ok=True)
            
            style_interpolation_experiment(
                cav_transfer,
                basis_result,
                content_imgs[0],  # Use first content image
                styles[0],        # Use first style image of this class
                output_subdir
            )
    
    logging.info("Multi-style experiment completed successfully!")