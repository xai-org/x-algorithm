export default function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-void">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.035)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,black_10%,transparent_75%)]" />

      <div className="absolute left-1/2 top-[-20%] h-[70vh] w-[70vh] -translate-x-1/2 animate-aurora rounded-full bg-besc-600/25 blur-[140px]" />
      <div className="absolute right-[-10%] top-[8%] h-[50vh] w-[50vh] animate-aurora rounded-full bg-besc-300/15 blur-[140px] [animation-delay:-8s]" />
      <div className="absolute bottom-[-15%] left-[-5%] h-[55vh] w-[55vh] animate-aurora rounded-full bg-volt/10 blur-[160px] [animation-delay:-14s]" />

      <div className="noise-overlay absolute inset-0 mix-blend-overlay" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-void" />
    </div>
  );
}
