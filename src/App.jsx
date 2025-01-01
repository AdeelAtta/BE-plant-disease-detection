import React, { useState } from 'react';
import { Upload, Image as LucideImage } from 'lucide-react';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await fetch('https://be-plant-disease-detection-production.up.railway.app/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-bold mb-4">Plant Disease Detection</h1>
        
        {/* Upload Section */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-gray-400 transition-colors">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
            id="imageInput"
          />
          <label htmlFor="imageInput" className="cursor-pointer">
            <div className="flex flex-col items-center space-y-2">
              <Upload className="h-8 w-8 text-gray-500" />
              <span className="text-sm text-gray-500">Click to upload an image</span>
            </div>
          </label>
        </div>

        {/* Preview */}
        {preview && (
          <div className="mt-6">
            <img 
              src={preview} 
              alt="Preview" 
              className="max-h-64 mx-auto rounded-lg shadow-md"
            />
          </div>
        )}

        {/* Detect Button */}
        <button
          onClick={handleSubmit}
          disabled={!selectedImage || loading}
          className={`mt-4 px-6 py-2 rounded-full ${
            !selectedImage || loading
              ? 'bg-gray-300 cursor-not-allowed'
              : 'bg-blue-500 hover:bg-blue-600'
          } text-white font-semibold transition-colors`}
        >
          {loading ? 'Analyzing...' : 'Detect Disease'}
        </button>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {/* Results */}
        {prediction && (
  <div className="mt-6">
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-2">Detection Result</h2>
      <p className="text-lg">
        <span className="font-medium">Disease:</span>{' '}
        {prediction.predicted_class.replace(/_/g, ' ')}
      </p>
      <p className="text-sm text-gray-600">
        Confidence: {prediction.confidence}%
      </p>
    </div>
  </div>
)}
      </div>
    </div>
  );
};

export default App;