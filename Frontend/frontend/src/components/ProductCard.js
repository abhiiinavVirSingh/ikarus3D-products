import React from 'react';
import './ProductCard.css'; // For styling

function ProductCard({ product }) {
  const { metadata, generated_blurb, score } = product;
  // Get the first image URL, or use a placeholder
  const imageUrl = metadata?.images?.length > 0 ? metadata.images[0] : 'https://via.placeholder.com/150';

  return (
    <div className="product-card">
      <img src={imageUrl} alt={metadata?.title || 'Product Image'} onError={(e) => { e.target.onerror = null; e.target.src='https://via.placeholder.com/150'; }}/>
      <div className="product-info">
        <h3>{metadata?.title || 'No Title'}</h3>
        <p><strong>Brand:</strong> {metadata?.brand || 'N/A'}</p>
        {metadata?.price && <p><strong>Price:</strong> ${metadata.price.toFixed(2)}</p>}
        {metadata?.color && <p><strong>Color:</strong> {metadata.color}</p>}
        {metadata?.material && <p><strong>Material:</strong> {metadata.material}</p>}
        <p className="blurb"><em>{generated_blurb || 'No description available.'}</em></p>
        {/* Optional: Display score or other metadata */}
        {/* <p><small>Similarity Score: {score?.toFixed(3)}</small></p> */}
      </div>
    </div>
  );
}

export default ProductCard;