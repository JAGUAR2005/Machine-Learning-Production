import { motion } from 'framer-motion';

interface AnimatedTitleProps {
  text: string;
  className?: string;
  delay?: number;
}

const AnimatedTitle = ({ text, className = "", delay = 0 }: AnimatedTitleProps) => (
  <motion.span className={className}>
    {text.split('').map((char, i) => (
      <motion.span
        key={i}
        className="char"
        initial={{ opacity: 0, y: 30, rotateX: -60 }}
        animate={{ opacity: 1, y: 0, rotateX: 0 }}
        transition={{ duration: 0.5, delay: delay + i * 0.04, ease: [0.22, 1, 0.36, 1] }}
        style={{ display: 'inline-block', whiteSpace: char === ' ' ? 'pre' : undefined }}
      >
        {char}
      </motion.span>
    ))}
  </motion.span>
);

export default AnimatedTitle;
