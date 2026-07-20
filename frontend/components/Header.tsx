export default function Header() {
  return (
    <header className="flex flex-col items-center gap-3 text-center">
      <h1 className="text-3xl font-semibold text-white sm:text-4xl">Image Captioner</h1>
      <p className="max-w-xl text-sm leading-relaxed text-slate-400 sm:text-base">
        Drop an image and generate a screen-reader-ready caption using a CLIP-encoder
        and a attention-driven GRU captioning model.
      </p>
    </header>
  );
}
