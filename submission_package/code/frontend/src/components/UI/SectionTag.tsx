interface SectionTagProps {
  label: string;
  number?: string;
}

const SectionTag = ({ label, number = "26" }: SectionTagProps) => (
  <span className="section-tag">{label}.{number}</span>
);

export default SectionTag;
