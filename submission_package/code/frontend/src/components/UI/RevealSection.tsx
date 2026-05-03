import React, { useRef } from 'react';
import { motion, useInView } from 'framer-motion';

interface RevealSectionProps {
  children: React.ReactNode;
  className?: string;
  id?: string;
}

const RevealSection = ({ children, className = "", id = "" }: RevealSectionProps) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.div
      ref={ref}
      id={id}
      className={className}
      initial={{ opacity: 0, y: 50 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
};

export default RevealSection;
