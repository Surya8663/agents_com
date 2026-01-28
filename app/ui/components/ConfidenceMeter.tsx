import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  TrendingUp, 
  TrendingDown,
  Zap
} from 'lucide-react';

interface ConfidenceMeterProps {
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  showDetails?: boolean;
  animated?: boolean;
  className?: string;
  onHoverChange?: (isHovering: boolean) => void;
}

const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  confidence,
  size = 'md',
  showLabel = true,
  showDetails = false,
  animated = true,
  className = '',
  onHoverChange
}) => {
  const [isHovering, setIsHovering] = useState(false);
  const [displayConfidence, setDisplayConfidence] = useState(0);
  
  // Size configurations
  const sizeConfig = {
    sm: { height: 8, fontSize: 'text-xs', iconSize: 12 },
    md: { height: 12, fontSize: 'text-sm', iconSize: 16 },
    lg: { height: 16, fontSize: 'text-base', iconSize: 20 }
  }[size];

  // Color based on confidence level
  const getColorConfig = (conf: number) => {
    if (conf >= 0.85) {
      return {
        gradient: 'from-green-500 to-emerald-600',
        bg: 'bg-green-100',
        text: 'text-green-800',
        icon: CheckCircle,
        label: 'High Confidence',
        description: 'Highly reliable extraction',
        recommendation: 'Automatically approved'
      };
    } else if (conf >= 0.65) {
      return {
        gradient: 'from-yellow-500 to-amber-600',
        bg: 'bg-yellow-100',
        text: 'text-yellow-800',
        icon: Zap,
        label: 'Medium Confidence',
        description: 'Generally reliable, minor uncertainty',
        recommendation: 'Consider spot checking'
      };
    } else if (conf >= 0.45) {
      return {
        gradient: 'from-orange-500 to-red-500',
        bg: 'bg-orange-100',
        text: 'text-orange-800',
        icon: Info,
        label: 'Low Confidence',
        description: 'Significant uncertainty detected',
        recommendation: 'Human review recommended'
      };
    } else {
      return {
        gradient: 'from-red-500 to-rose-700',
        bg: 'bg-red-100',
        text: 'text-red-800',
        icon: AlertTriangle,
        label: 'Very Low Confidence',
        description: 'High risk of error or hallucination',
        recommendation: 'Immediate human review required'
      };
    }
  };

  // Smooth animation for confidence value
  useEffect(() => {
    if (!animated) {
      setDisplayConfidence(confidence);
      return;
    }

    const timer = setTimeout(() => {
      const step = 0.02;
      const diff = confidence - displayConfidence;
      
      if (Math.abs(diff) > step) {
        setDisplayConfidence(prev => prev + (diff > 0 ? step : -step));
      } else {
        setDisplayConfidence(confidence);
      }
    }, 16); // ~60fps

    return () => clearTimeout(timer);
  }, [confidence, displayConfidence, animated]);

  const colorConfig = getColorConfig(displayConfidence);
  const Icon = colorConfig.icon;
  const percentage = Math.round(displayConfidence * 100);

  // Get trend indicator
  const getTrendIndicator = () => {
    if (confidence > 0.7) {
      return {
        icon: TrendingUp,
        color: 'text-green-600',
        label: 'Improving'
      };
    } else if (confidence < 0.4) {
      return {
        icon: TrendingDown,
        color: 'text-red-600',
        label: 'Declining'
      };
    }
    return null;
  };

  const trend = getTrendIndicator();

  return (
    <div 
      className={`relative ${className}`}
      onMouseEnter={() => {
        setIsHovering(true);
        onHoverChange?.(true);
      }}
      onMouseLeave={() => {
        setIsHovering(false);
        onHoverChange?.(false);
      }}
    >
      {/* Main meter */}
      <div className="space-y-2">
        {showLabel && (
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Icon className="w-4 h-4" />
              <span className={`font-medium ${colorConfig.text}`}>
                {colorConfig.label}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className={`font-bold ${sizeConfig.fontSize}`}>
                {percentage}%
              </span>
              {trend && (
                <div className={`flex items-center ${trend.color}`}>
                  {trend && trend.icon && <trend.icon className="w-4 h-4" />}
                  <span className="text-xs ml-1">{trend.label}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Progress bar */}
        <div className="relative">
          {/* Background track */}
          <div 
            className="w-full bg-gray-200 rounded-full overflow-hidden"
            style={{ height: sizeConfig.height }}
          >
            {/* Gradient fill */}
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${percentage}%` }}
              transition={{ 
                type: "spring", 
                stiffness: 100, 
                damping: 20,
                delay: 0.1 
              }}
              className={`h-full rounded-full bg-linear-to-r ${colorConfig.gradient}`}
            />
            
            {/* Confidence markers */}
            <div className="absolute inset-0 flex justify-between px-1">
              {[0, 25, 50, 75, 100].map((mark) => (
                <div
                  key={mark}
                  className="w-0.5 h-full bg-white/30"
                  style={{ marginLeft: `${mark}%` }}
                />
              ))}
            </div>
          </div>

          {/* Threshold indicators */}
          <div className="absolute top-0 h-full flex justify-between w-full pointer-events-none">
            {[
              { value: 0.45, label: 'Review', color: 'border-red-400' },
              { value: 0.65, label: 'Check', color: 'border-yellow-400' },
              { value: 0.85, label: 'Auto', color: 'border-green-400' }
            ].map((threshold) => (
              <div
                key={threshold.label}
                className="relative"
                style={{ left: `${threshold.value * 100}%` }}
              >
                <div className={`absolute top-0 w-0.5 h-full border-l-2 ${threshold.color}`} />
                <div className="absolute top-0 transform -translate-x-1/2 -translate-y-6">
                  <div className="text-xs text-gray-500 whitespace-nowrap">
                    {threshold.label}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Detailed hover card */}
      <AnimatePresence>
        {isHovering && showDetails && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="absolute z-50 mt-2 w-64 p-4 bg-white rounded-xl shadow-xl border"
            style={{ left: '50%', transform: 'translateX(-50%)' }}
          >
            {/* Arrow */}
            <div className="absolute -top-2 left-1/2 transform -translate-x-1/2">
              <div className="w-4 h-4 bg-white border-t border-l transform rotate-45" />
            </div>

            <div className="space-y-3">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <div className={`p-2 rounded-lg ${colorConfig.bg}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-900">
                        {colorConfig.label}
                      </h4>
                      <p className="text-sm text-gray-600">
                        {colorConfig.description}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {percentage}%
                </div>
              </div>

              {/* Recommendation */}
              <div className="p-3 rounded-lg bg-gray-50">
                <div className="text-sm font-medium text-gray-700 mb-1">
                  Recommendation
                </div>
                <div className="text-sm text-gray-600">
                  {colorConfig.recommendation}
                </div>
              </div>

              {/* Confidence breakdown (simulated) */}
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-700">
                  Confidence Factors
                </div>
                {[
                  { label: 'OCR Quality', value: 0.8, weight: 0.15 },
                  { label: 'Layout Detection', value: 0.7, weight: 0.1 },
                  { label: 'Semantic Understanding', value: 0.9, weight: 0.2 },
                  { label: 'Multi-Modal Fusion', value: 0.75, weight: 0.15 },
                ].map((factor, index) => (
                  <div key={index} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">{factor.label}</span>
                      <span className="font-medium">
                        {Math.round(factor.value * 100)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-linear-to-r from-blue-500 to-cyan-500 rounded-full"
                        style={{ width: `${factor.value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Action buttons */}
              <div className="pt-2 flex space-x-2">
                <button className="flex-1 py-2 text-sm bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors">
                  View Details
                </button>
                {confidence < 0.65 && (
                  <button className="flex-1 py-2 text-sm bg-amber-50 text-amber-600 rounded-lg hover:bg-amber-100 transition-colors">
                    Send for Review
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Click overlay for details */}
      {isHovering && showDetails && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setIsHovering(false)}
        />
      )}
    </div>
  );
};

export default ConfidenceMeter;